# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""

DataWig Imputer:
Imputes missing values in tables

"""

import os
import glob
import pickle
import time
import itertools
import inspect
from typing import List, Any, Tuple
import mxnet as mx
from mxnet.callback import save_checkpoint
import pandas as pd
import numpy as np

from .mxnet_output_symbols import make_categorical_loss, make_numerical_loss
from .utils import timing, MeanSymbol, LogMetricCallBack, logger, \
    random_split, AccuracyMetric, gpu_device, ColumnOverwriteException, merge_dicts
from .column_encoders import ColumnEncoder, NumericalEncoder, CategoricalEncoder
from .iterators import ImputerIterDf
from .mxnet_input_symbols import Featurizer, ImageFeaturizer
from .evaluation import evaluate_and_persist_metrics


class Imputer:
    """

    Imputer model based on deep learning trained with MxNet

    Given a data frame with string columns, a model is trained to predict observed values in one
    or more column using values observed in other columns. The model can then be used to impute
    missing values.

    :param data_encoders: list of datawig.mxnet_input_symbol.ColumnEncoders,
                            output_column name must match field_name of data_featurizers
    :param data_featurizers: list of Featurizer;
    :param label_encoders: list of CategoricalEncoder or NumericalEncoder
    :param output_path: path to store model and metrics


    """

    def __init__(self,
                 data_encoders: List[ColumnEncoder],
                 data_featurizers: List[Featurizer],
                 label_encoders: List[ColumnEncoder],
                 output_path="") -> None:

        self.ctx = None
        self.module = None
        self.data_encoders = data_encoders

        self.batch_size = 16

        self.data_featurizers = data_featurizers
        self.label_encoders = label_encoders
        self.final_fc_hidden_units = []

        self.train_losses = None
        self.test_losses = None

        self.training_time = 0.

        self.precision_recall_curves = {}

        if len(self.data_featurizers) != len(self.data_encoders):
            raise ValueError("Argument Number of data_featurizers ({}) \
                              must match number of data_encoders ({})".format(len(self.data_encoders), len(self.data_featurizers)))

        for encoder in self.data_encoders:
            encoder_type = type(encoder)
            if not issubclass(encoder_type, ColumnEncoder):
                raise ValueError("Arguments passed as data_encoder must be valid " +
                                 "datawig.column_encoders.ColumnEncoder, was {}".format(
                                     encoder_type))

        for encoder in self.label_encoders:
            encoder_type = type(encoder)
            if encoder_type not in [CategoricalEncoder, NumericalEncoder]:
                raise ValueError("Arguments passed as label_columns must be \
                                 datawig.column_encoders.CategoricalEncoder or NumericalEncoder, \
                                 was {}".format(encoder_type))

        encoder_outputs = [encoder.output_column for encoder in self.data_encoders]

        for featurizer in self.data_featurizers:
            featurizer_type = type(featurizer)
            if not issubclass(featurizer_type, Featurizer):
                raise ValueError("Arguments passed as data_featurizers must be valid \
                                 datawig.mxnet_input_symbols.Featurizer type, \
                                 was {}".format(featurizer_type))

            if featurizer.field_name not in encoder_outputs:
                raise ValueError(
                    "List of encoder outputs [{}] does not contain featurizer input for {}".format(
                        ", ".join(encoder_outputs), featurizer_type))
            # TODO: check whether encoder type matches requirements of featurizer

        # collect names of data and label columns
        input_col_names = [c.field_name for c in self.data_featurizers]
        label_col_names = list(itertools.chain(*[c.input_columns for c in self.label_encoders]))

        if len(set(input_col_names).intersection(set(label_col_names))) != 0:
            raise ValueError("cannot train with label columns that are in the input")

        # if there is no output directory provided, try to write to current dir
        if (output_path == '') or (not output_path):
            output_path = '.'

        self.output_path = output_path

        # if there was no output dir provided, name it to the label (-list) fitted
        if self.output_path == ".":
            label_names = [c.output_column.lower().replace(" ", "_") for c in self.label_encoders]
            self.output_path = "-".join(label_names)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        self.module_path = os.path.join(self.output_path, "model")

        self.metrics_path = os.path.join(self.output_path, "fit-test-metrics.json")

    def __check_data(self, data_frame: pd.DataFrame) -> None:
        """
        Checks some aspects of data quality, currently just the label distribution

        Currently checked are:
         - label overlap in training and test data

        if these requirements are not met a warning is raised.

        TODO: more data quality checks; note that in order to avoid unneccessary passes through
        data, some warnings are raised in CategoricalEncoder.fit, too.

        """
        for col_enc in self.label_encoders:

            if not col_enc.is_fitted():
                logger.warning(
                    "Data encoder {} for columns {} is not fitted yet, cannot check data".format(
                        type(col_enc), ", ".join(col_enc.input_columns)))
            elif isinstance(col_enc, CategoricalEncoder):
                values_not_in_test_set = set(col_enc.token_to_idx.keys()) - \
                                         set(data_frame[col_enc.input_columns[0]].unique())
                logger.warning(
                    "Test set does not contain any ocurrences of values [{}] in column [{}], "
                    "consider using a more representative test set.".format(
                        ", ".join(values_not_in_test_set),
                        col_enc.input_columns[0]))

    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            ctx: mx.context = mx.gpu() if gpu_device() else mx.cpu(),
            learning_rate: float = 1e-3,
            num_epochs: int = 100,
            patience: int = 3,
            test_split: float = .1,
            weight_decay: float = 0.,
            batch_size: int = 16,
            final_fc_hidden_units: List[int] = None):
        """
        Trains and stores imputer model

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, [test_split] % of the training
                        data are used as test data
        :param ctx: mxnet context (default mx.cpu())
        :param learning_rate: learning rate for stochastic gradient descent (default 1e-4)
        :param num_epochs: maximal number of training epochs (default 100)
        :param patience: used for early stopping; after [patience] epochs with no improvement,
                        training is stopped. (default 3)
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                        separate for determining model convergence
        :param weight_decay: regularizer (default 0)
        :batch_size: default 16
        :param final_fc_hidden_units: list of dimensions for the final fully connected layer.
        :return: trained imputer model
        """
        if not final_fc_hidden_units:
            final_fc_hidden_units = []

        # make sure the output directory is writable
        assert os.access(self.output_path, os.W_OK), "Cannot write to directory {}".format(
            self.output_path)

        self.batch_size = batch_size
        self.final_fc_hidden_units = final_fc_hidden_units

        self.ctx = ctx

        if (train_df is None) or (not isinstance(train_df, pd.core.frame.DataFrame)):
            raise ValueError("Need a non-empty DataFrame for fitting Imputer model")

        if test_df is None:
            train_df, test_df = random_split(train_df, [1.0 - test_split, test_split])

        iter_train, iter_test = self.__build_iterators(train_df, test_df, test_split)

        self.__check_data(test_df)

        self.__build_module(iter_train)
        self.__fit_module(iter_train, iter_test, learning_rate, num_epochs, patience, weight_decay)
        _, metrics = self.__transform_and_compute_metrics_mxnet_iter(iter_test,
                                                                     metrics_path=self.metrics_path)

        for att, att_metric in metrics.items():
            if isinstance(att_metric, dict) and ('precision_recall_curves' in att_metric):
                self.precision_recall_curves[att] = att_metric['precision_recall_curves']

        self.__prune_models()
        self.save()

        return self

    def __fit_module(self,
                     iter_train: ImputerIterDf,
                     iter_test: ImputerIterDf,
                     learning_rate: float,
                     num_epochs: int,
                     patience: int,
                     weight_decay: float) -> None:
        """

        Trains the mxnet module

        :param learning_rate: learning rate of sgd
        :param num_epochs: maximal number of training epochs
        :param patience: early stopping
        :param weight_decay: regularizer

        """
        metric_name = 'cross-entropy'

        train_cb = LogMetricCallBack([metric_name])
        test_cb = LogMetricCallBack([metric_name], patience=patience)

        def checkpoint(epoch, sym, arg, aux):
            save_checkpoint(self.module_path, epoch, sym, arg, aux)

        start = time.time()
        cross_entropy_metric = MeanSymbol(metric_name)
        accuracy_metrics = [
            AccuracyMetric(name=label_col.output_column, label_index=i) for i, label_col in
            enumerate(self.label_encoders)
        ]
        combined_metric = mx.metric.CompositeEvalMetric(
            metrics=[cross_entropy_metric] + accuracy_metrics)

        image_network_params = {}
        for featurizer in self.data_featurizers:
            if isinstance(featurizer, ImageFeaturizer):
                image_network_params = featurizer.params

        with timing("fit model"):
            try:
                self.module.fit(
                    train_data=iter_train,
                    eval_data=iter_test,
                    eval_metric=combined_metric,
                    num_epoch=num_epochs,
                    initializer=mx.initializer.Mixed(['image_featurizer', '.*'],
                                                     [mx.init.Load(image_network_params),
                                                      mx.init.Xavier(factor_type="in",
                                                                     magnitude=2.34)]),
                    optimizer='adam',
                    optimizer_params=(('learning_rate', learning_rate), ('wd', weight_decay)),
                    batch_end_callback=[train_cb,
                                        mx.callback.Speedometer(iter_train.batch_size, 20,
                                                                auto_reset=True)],
                    eval_end_callback=test_cb,
                    epoch_end_callback=checkpoint
                )
            except StopIteration:
                # catch the StopIteration exception thrown when early stopping condition is reached
                # this is ugly but the only way to use module api and have early stopping
                logger.info("Stopping training, patience reached")
                pass

        self.training_time = time.time() - start
        self.train_losses, self.test_losses = train_cb.metrics[metric_name], test_cb.metrics[
            metric_name]

    def __build_module(self, iter_train: ImputerIterDf) -> None:

        # construct the losses
        predictions, loss = self.__make_loss()

        logger.info("Building output symbols")
        output_symbols = []
        for col_enc, output in zip(self.label_encoders, predictions):
            output_symbols.append(
                mx.sym.BlockGrad(output, name="pred-{}".format(col_enc.output_column)))

        # Get params to fix assuming we don't want to fine tune
        fine_tune = any([isinstance(feat, ImageFeaturizer) for feat in self.data_featurizers])

        fixed_params = []
        if not fine_tune:
            for name in loss.list_arguments():
                if "image_featurizer" in name:
                    fixed_params.append(name)
        else:
            fixed_params = None

        self.module = mx.mod.Module(
            mx.sym.Group([loss] + output_symbols),
            context=self.ctx,
            data_names=[name for name, dim in iter_train.provide_data],
            label_names=[name for name, dim in iter_train.provide_label],
            fixed_param_names=fixed_params
        )

        self.module.bind(data_shapes=iter_train.provide_data, label_shapes=iter_train.provide_label)

    def __build_iterators(self,
                          train_df: pd.DataFrame,
                          test_df: pd.DataFrame,
                          test_split: float) -> Tuple[ImputerIterDf, ImputerIterDf]:
        """

        Builds iterators from data frame

        :param train_df: training data as pandas DataFrame
        :param test_df: test data, can be None
        :param test_split: test data split, used if test_df is None
        :return: train and test iterators for mxnet model

        """

        # if this is a pandas df
        if not isinstance(train_df, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if (test_df is not None) & (not isinstance(test_df, pd.core.frame.DataFrame)):
            raise ValueError("Only pandas data frames are supported")

        input_col_names = [c.field_name for c in self.data_featurizers]
        label_col_names = list(itertools.chain(*[c.input_columns for c in self.label_encoders]))

        missing_columns = set(input_col_names + label_col_names) - set(train_df.columns)

        if len(missing_columns) > 0:
            ValueError("Training DataFrame does not contain required column {}".format(missing_columns))

        for encoder in self.label_encoders:
            if not encoder.is_fitted():
                encoder_type = type(encoder)
                logger.info("Fitting label encoder {} on {} rows \
                            of training data".format(encoder_type, len(train_df)))
                encoder.fit(train_df)

        # discard all rows which contain labels that will not be learned/predicted
        train_df = self.__drop_missing_labels(train_df, how='all')

        # if there is no test data set provided, split one off the training data
        if test_df is None:
            train_df, test_df = random_split(train_df, [1.0 - test_split, test_split])

        test_df = self.__drop_missing_labels(test_df, how='all')

        logger.info("Train: {}, Test: {}".format(len(train_df), len(test_df)))

        for encoder in self.data_encoders:
            if not encoder.is_fitted():
                encoder_type = type(encoder)
                logger.info(
                    "Fitting data encoder {} on columns {} and {} rows of training data".format(
                        encoder_type, ", ".join(encoder.input_columns), len(train_df)))

                encoder.fit(train_df)

        logger.info("Building Train Iterator with {} elements".format(len(train_df)))
        iter_train = ImputerIterDf(
            data_frame=train_df,
            data_columns=self.data_encoders,
            label_columns=self.label_encoders,
            batch_size=self.batch_size
        )

        logger.info("Building Test Iterator with {} elements".format(len(test_df)))
        iter_test = ImputerIterDf(
            data_frame=test_df,
            data_columns=iter_train.data_columns,
            label_columns=iter_train.label_columns,
            batch_size=self.batch_size
        )

        return iter_train, iter_test

    def __make_loss(self, eps: float = 1e-5) -> Tuple[Any, Any]:

        logger.info("Concatenating all {} latent symbols".format(len(self.data_featurizers)))

        unique_input_field_names = set([feat.field_name for feat in self.data_featurizers])
        if len(unique_input_field_names) < len(self.data_featurizers):
            raise ValueError("Input fields of Featurizers outputs of ColumnEncoders must be unique but \
                there were duplicates in {}, consider \
                explicitly providing output column names to ColumnEncoders".format(", ".join(unique_input_field_names)))

        # construct mxnet symbols for the data columns
        latents = mx.sym.concat(*[f.latent_symbol() for f in self.data_featurizers], dim=1)

        # build predictions and loss for each single output
        outputs = []
        for output_col in self.label_encoders:
            if isinstance(output_col, CategoricalEncoder):
                logger.info("Constructing categorical loss for column {} and {} labels".format(
                    output_col.output_column, output_col.max_tokens))
                outputs.append(make_categorical_loss(latents, output_col.output_column,
                                                     output_col.max_tokens + 1,
                                                     self.final_fc_hidden_units))
            elif isinstance(output_col, NumericalEncoder):
                logger.info(
                    "Constructing numerical loss for column {}".format(output_col.output_column))
                outputs.append(make_numerical_loss(latents, output_col.output_column))

        predictions, losses = zip(*outputs)

        # compute mean loss for each output
        mean_batch_losses = [mx.sym.mean(l) + eps for l in losses]

        # normalize the loss contribution of each label by the mean over the batch
        normalized_losses = [mx.sym.broadcast_div(l, mean_loss) for l, mean_loss in
                             zip(losses, mean_batch_losses)]

        # multiply the loss by the mean of all losses of all labels to preserve the gradient norm
        mean_label_batch_loss = mx.sym.ElementWiseSum(*mean_batch_losses) / float(
            len(mean_batch_losses))

        # normalize batch
        loss = mx.sym.broadcast_mul(
            mx.sym.ElementWiseSum(*normalized_losses) / float(len(mean_batch_losses)),
            mean_label_batch_loss
        )
        loss = mx.sym.MakeLoss(loss, normalization='valid', valid_thresh=1e-6)

        return predictions, loss

    def __transform_mxnet_iter(self, mxnet_iter: ImputerIterDf) -> dict:
        """

        Imputes values given an mxnet iterator (see iterators)

        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': list} where list contains the string predictions

        """
        labels, model_outputs = zip(*self.__predict_mxnet_iter(mxnet_iter).items())
        predictions = {}
        for col_enc, label, model_output in zip(mxnet_iter.label_columns, labels, model_outputs):
            if isinstance(col_enc, CategoricalEncoder):
                predictions[label] = col_enc.decode(pd.Series(model_output.argmax(axis=1)))
            elif isinstance(col_enc, NumericalEncoder):
                predictions[label] = model_output
        return predictions

    @staticmethod
    def __filter_predictions(predictions: dict,
                             precision_threshold: float,
                             precision_recall_curve: dict) -> dict:
        """
        Filters predictions below precision threshold

        :param predictions:  predictions
        :param precision_threshold: precision threshold
        :param precision_recall_curve: precision recall curves as dict
        :return: filtered predictions
        """
        filtered_predictions = []
        for prediction in predictions:
            filtered_prediction = ()
            label, score = prediction[0]
            if precision_threshold > 0.0:
                if label in precision_recall_curve:
                    test_precisions = precision_recall_curve[label]['precision']
                    test_thresholds = precision_recall_curve[label]['thresholds']
                    n_labels = len(test_precisions)
                    # find threshold such that prediction is above precision threshold also for
                    # multimodal distribution
                    below_threshold = np.where(test_precisions[::-1] < precision_threshold)[0]
                    if len(below_threshold) < n_labels:
                        if len(below_threshold) == 0:
                            threshold_idx = 0
                        else:
                            threshold_idx = n_labels - (below_threshold[0] + 1)
                        score_threshold = test_thresholds[threshold_idx]
                        if score >= score_threshold:
                            filtered_prediction = (label, score)
                else:
                    logger.warning("Label {} not found in test set, discarding prediction \
                                   ({}, {}), consider setting precision_threshold to \
                                   0.0 for obtaining these predictions".format(label, label, score))
            else:
                filtered_prediction = (label, score)
            filtered_predictions.append(filtered_prediction)
        return filtered_predictions

    def __predict_above_precision_mxnet_iter(self,
                                             mxnet_iter: ImputerIterDf,
                                             precision_threshold: float = 0.95) -> dict:
        """
        Imputes values only if predictions are above certain precision threshold,
            determined on test set during fit

        :param mxnet_iter: iterator, see ImputerIter in iterators.py
        :param precision_threshold: don't predict if predicted class probability is below
            this precision threshold
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels

        """
        predictions = self.__predict_top_k_mxnet_iter(mxnet_iter, top_k=1)
        for col_enc, att in zip(self.label_encoders, predictions.keys()):
            if isinstance(col_enc, CategoricalEncoder):
                predictions[att] = self.__filter_predictions(
                    predictions[att],
                    precision_threshold,
                    self.precision_recall_curves[att])
            else:
                logger.info("Precision filtering only for CategoricalEncoder returning \
                            {} unfiltered".format(att))
                predictions[att] = predictions[att]

        return predictions

    def __predict_mxnet_iter(self, mxnet_iter):
        """
        Returns the probabilities for each class
        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels
        """
        if not self.module.for_training:
            self.module.bind(data_shapes=mxnet_iter.provide_data,
                             label_shapes=mxnet_iter.provide_label)
        # FIXME: truncation to [:mxnet_iter.start_padding_idx] because of having to set
        # last_batch_handle to discard.
        # truncating bottom rows for each output module while preserving all the columns
        mod_output = [o.asnumpy()[:mxnet_iter.start_padding_idx, :] for o in
                      self.module.predict(mxnet_iter)[1:]]
        output = {}
        for label_encoder, pred in zip(mxnet_iter.label_columns, mod_output):
            if isinstance(label_encoder, NumericalEncoder):
                output[label_encoder.output_column] = label_encoder.decode(pred)
            else:
                output[label_encoder.output_column] = pred
        return output

    def __predict_top_k_mxnet_iter(self, mxnet_iter, top_k=5):
        """
        For categorical outputs, returns tuples of (label, probability) for the top_k most likely
        predicted classes
        For numerical outputs, returns just prediction

        :param mxnet_iter:  iterator, see ImputerIter in iterators.py
        :return: dict of {'column_name': list} where list is a list of (label, probability)
                    tuples or numerical values

        """
        col_enc_att_probas = zip(mxnet_iter.label_columns,
                                 *zip(*self.__predict_mxnet_iter(mxnet_iter).items()))
        top_k_predictions = {}
        for col_enc, att, probas in col_enc_att_probas:
            if isinstance(col_enc, CategoricalEncoder):
                top_k_predictions[att] = []
                for pred in probas:
                    top_k_pred_idx = pred.argsort()[-top_k:][::-1]
                    label_proba_tuples = [(col_enc.decode_token(idx), pred[idx]) for idx in
                                          top_k_pred_idx]
                    top_k_predictions[att].append(label_proba_tuples)
            else:
                logger.info(
                    "Top-k only for CategoricalEncoder, dropping {}, {}".format(att, type(col_enc)))
                top_k_predictions[att] = probas

        return top_k_predictions

    def __transform_and_compute_metrics_mxnet_iter(self,
                                                   mxnet_iter: ImputerIterDf,
                                                   metrics_path: str = None) -> Tuple[dict, dict]:
        """

        Returns predictions and metrics (average and per class)

        :param mxnet_iter:
        :param metrics_path: if not None and exists, metrics are serialized as json to this path.
        :return: predictions and metrics

        """
        # get thresholded predictions for predictions and standard metrics
        mxnet_iter.reset()
        all_predictions = self.__transform_mxnet_iter(mxnet_iter)
        # reset iterator, compute probabilistic outputs for all classes for precision/recall curves
        mxnet_iter.reset()
        all_predictions_proba = self.__predict_mxnet_iter(mxnet_iter)
        mxnet_iter.reset()
        true_labels_idx_array = mx.nd.concat(
            *[mx.nd.concat(*[l for l in b.label], dim=1) for b in mxnet_iter], dim=0)
        true_labels_string = {}
        true_labels_idx = {}
        predictions_categorical = {}
        predictions_categorical_proba = {}

        predictions_numerical = {}
        true_labels_numerical = {}

        for l_idx, col_enc in enumerate(mxnet_iter.label_columns):
            n_predictions = len(all_predictions[col_enc.output_column])
            if isinstance(col_enc, CategoricalEncoder):
                predictions_categorical[col_enc.output_column] = all_predictions[
                    col_enc.output_column]
                predictions_categorical_proba[col_enc.output_column] = all_predictions_proba[
                    col_enc.output_column]
                attribute = col_enc.output_column
                true_labels_idx[attribute] = true_labels_idx_array[:n_predictions, l_idx].asnumpy()
                true_labels_string[attribute] = [col_enc.decode_token(idx) for idx in
                                                 true_labels_idx[attribute]]
            elif isinstance(col_enc, NumericalEncoder):
                true_labels_numerical[col_enc.output_column] = \
                    true_labels_idx_array[:n_predictions, l_idx].asnumpy()
                predictions_numerical[col_enc.output_column] = \
                    all_predictions[col_enc.output_column]

        metrics = evaluate_and_persist_metrics(true_labels_string,
                                               true_labels_idx,
                                               predictions_categorical,
                                               predictions_categorical_proba,
                                               metrics_path,
                                               None,
                                               true_labels_numerical,
                                               predictions_numerical)

        return merge_dicts(predictions_categorical, predictions_numerical), metrics

    def transform(self, data_frame: pd.DataFrame) -> dict:
        """
        Imputes values given an mxnet iterator (see iterators)
        :param data_frame:  pandas data frame (pandas)
        :return: dict of {'column_name': list} where list contains the string predictions
        """
        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        return self.__transform_mxnet_iter(mxnet_iter)

    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                imputation_suffix: str = "_imputed",
                score_suffix: str = "_imputed_proba") -> pd.DataFrame:
        """
        Computes imputations for numerical or categorical values

        For categorical imputations, most likely values are imputed if values are above a certain
        precision threshold computed on the validation set
        Precision is calculated as part of the `datawig.evaluate_and_persist_metrics` function.

        For numerical imputations, no thresholding is applied.

        Returns original dataframe with imputations and respective likelihoods as estimated by
        imputation model in additional columns; names of imputation columns are that of the label
        suffixed with `imputation_suffix`, names of respective likelihood columns are suffixed with
        `score_suffix`

        :param data_frame:   pandas data_frame
        :param precision_threshold: double between 0 and 1 indicating precision threshold for each
                                    imputation
        :param imputation_suffix: suffix for imputation columns
        :param score_suffix: suffix for imputation score columns
        :return: original dataframe with imputations and their likelihoods in additional columns
        """
        numerical_outputs = list(
            itertools.chain(
                *[c.input_columns for c in self.label_encoders if isinstance(c, NumericalEncoder)]))

        predictions = self.predict_above_precision(data_frame, precision_threshold).items()
        for label, imputations in predictions:
            imputation_col = label + imputation_suffix
            if data_frame.columns.contains(imputation_col):
                raise ColumnOverwriteException(
                    "DataFrame contains column {}; remove column and try again".format(
                        imputation_col))

            if label not in numerical_outputs:
                imputation_proba_col = label + score_suffix
                if data_frame.columns.contains(imputation_proba_col):
                    raise ColumnOverwriteException(
                        "DataFrame contains column {}; remove column and try again".format(
                            imputation_proba_col))

                imputed_values, imputed_value_scores = [], []
                for imputation_above_precision_threshold in imputations:
                    if not imputation_above_precision_threshold:
                        imputed_value, imputed_value_score = "", np.nan
                    else:
                        imputed_value, imputed_value_score = imputation_above_precision_threshold
                    imputed_values.append(imputed_value)
                    imputed_value_scores.append(imputed_value_score)

                data_frame[imputation_col] = imputed_values
                data_frame[imputation_proba_col] = imputed_value_scores

            elif label in numerical_outputs:
                data_frame[imputation_col] = imputations

        return data_frame

    def predict_proba(self, data_frame: pd.DataFrame) -> dict:
        """
        Returns the probabilities for each class
        :param data_frame:  data frame
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels
        """
        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        return self.__predict_mxnet_iter(mxnet_iter)

    def predict_above_precision(self, data_frame: pd.DataFrame, precision_threshold=0.95) -> dict:
        """

        Returns the probabilities for each class

        :param data_frame:  data frame
        :param precision_threshold: don't predict if predicted class probability is below this
                                        precision threshold
        :return: dict of {'column_name': array}, array is a numpy array of shape samples-by-labels

        """
        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        return self.__predict_above_precision_mxnet_iter(mxnet_iter,
                                                         precision_threshold=precision_threshold)

    def predict_proba_top_k(self, data_frame: pd.DataFrame, top_k: int = 5) -> dict:
        """

        Returns tuples of (label, probability) for the top_k most likely predicted classes

        :param data_frame:  pandas data frame
        :param top_k: number of most likely predictions to return
        :return: dict of {'column_name': list} where list is a list of (label, probability) tuples

        """
        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        return self.__predict_top_k_mxnet_iter(mxnet_iter, top_k)

    def transform_and_compute_metrics(self, data_frame: pd.DataFrame, metrics_path=None) -> dict:
        """

        Returns predictions and metrics (average and per class)

        :param data_frame:  data frame
        :param metrics_path: if not None and exists, metrics are serialized as json to this path.
        :return:
        """
        for col_enc in self.label_encoders:
            if col_enc.output_column not in data_frame.columns:
                raise ValueError(
                    "Cannot compute metrics: Label Column {} not found in \
                    input DataFrame with columns {}".format(col_enc.output_column, ", ".join(df.columns)))

        mxnet_iter = self.__mxnet_iter_from_df(data_frame)
        return self.__transform_and_compute_metrics_mxnet_iter(mxnet_iter,
                                                               metrics_path=metrics_path)

    def __drop_missing_labels(self, data_frame: pd.DataFrame, how='all') -> pd.DataFrame:
        """

        Drops rows of data frame that contain missing labels

        :param data_frame: pandas data frame
        :param how: ['all', 'any'] whether to drop rows if all labels are missing or if just
                    any label is missing
        :return: pandas DataFrame

        """
        n_samples = len(data_frame)
        missing_idx = None
        for col_enc in self.label_encoders:
            if isinstance(col_enc, CategoricalEncoder):
                # for CategoricalEncoders, exclude rows that are either nan or not in the
                # token_to_idx mapping
                col_missing_idx = data_frame[col_enc.input_columns[0]].isna() | \
                                  ~data_frame[col_enc.input_columns[0]].isin(
                                      col_enc.token_to_idx.keys())
            elif isinstance(col_enc, NumericalEncoder):
                # for NumericalEncoders, exclude rows that are nan
                col_missing_idx = data_frame[col_enc.input_columns[0]].isna()

            logger.info("Detected {} rows with missing labels \
                        for column {}".format(col_missing_idx.sum(), col_enc.input_columns[0]))

            if missing_idx is None:
                missing_idx = col_missing_idx
            elif how == 'all':
                missing_idx = missing_idx & col_missing_idx
            elif how == 'any':
                missing_idx = missing_idx | col_missing_idx

        logger.info("Dropping {}/{} rows".format(missing_idx.sum(), n_samples))

        return data_frame.loc[~missing_idx, :]

    def __prune_models(self):
        """

        Removes all suboptimal models from output directory

        """
        best_model = glob.glob(self.module_path + "*{}.params".format(self.__get_best_epoch()))
        logger.info("Keeping {}".format(best_model[0]))
        worse_models = set(glob.glob(self.module_path + "*.params")) - set(best_model)
        # remove worse models
        for worse_epoch in worse_models:
            logger.info("Deleting {}".format(worse_epoch))
            os.remove(worse_epoch)

    def __get_best_epoch(self):
        """

        Retrieves the best epoch, i.e. the minimum of the test_losses

        :return: best epoch
        """
        return sorted(enumerate(self.test_losses), key=lambda x: x[1])[0][0]

    def save(self):
        """

        Saves model to disk, except mxnet module which is stored separately during fit

        """
        # save all params but the mxnet module
        params = {k: v for k, v in self.__dict__.items() if k != 'module'}
        pickle.dump(params, open(os.path.join(self.output_path, "imputer.pickle"), "wb"))

    @staticmethod
    def load(output_path: str) -> Any:
        """

        Loads model from output path

        :param output_path: output_path field of trained Imputer model
        :return: imputer model

        """

        logger.info("Output path for loading Imputer {}".format(output_path))
        params = pickle.load(open(os.path.join(output_path, "imputer.pickle"), "rb"))
        imputer_signature = inspect.getfullargspec(Imputer.__init__)[0]
        # get constructor args
        constructor_args = {p: params[p] for p in imputer_signature if p != 'self'}
        non_constructor_args = {p: params[p] for p in params.keys() if
                                p not in ['self'] + list(constructor_args.keys())}

        # use all relevant fields to instantiate Imputer
        imputer = Imputer(**constructor_args)
        # then set all other args
        for arg, value in non_constructor_args.items():
            setattr(imputer, arg, value)

        # the module path must be updated when loading the Imputer, too
        imputer.module_path = os.path.join(output_path, 'model')
        imputer.output_path = output_path
        ctx = imputer.ctx

        logger.info("Loading mxnet model from {}".format(imputer.module_path))
        # deserialize mxnet module
        imputer.module = mx.module.Module.load(
            imputer.module_path,
            imputer.__get_best_epoch(),
            context=ctx,
            data_names=[s.field_name for s in imputer.data_featurizers],
            label_names=[s.output_column for s in imputer.label_encoders]
        )
        return imputer

    def __mxnet_iter_from_df(self, data_frame: pd.DataFrame) -> ImputerIterDf:
        """

        Transforms dataframe into imputer iterator for mxnet

        :param data_frame: pandas DataFrame
        :return: ImputerIterDf
        """
        return ImputerIterDf(
            data_frame=data_frame,
            data_columns=self.data_encoders,
            label_columns=self.label_encoders,
            batch_size=self.batch_size
        )
