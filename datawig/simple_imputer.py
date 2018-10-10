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

DataWig SimpleImputer:
Uses some simple default encoders and featurizers that usually yield decent imputation quality

"""
import pickle
import os
import json
import inspect
from typing import List, Dict, Any
import itertools
import mxnet as mx
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .utils import logger, get_context, random_split, rand_string
from .imputer import Imputer
from .column_encoders import BowEncoder, CategoricalEncoder, NumericalEncoder, ColumnEncoder
from .mxnet_input_symbols import BowFeaturizer, NumericalFeaturizer, Featurizer


class SimpleImputer:
    """

    SimpleImputer model based on n-grams of concatenated strings of input columns and concatenated
    numerical features, if provided.

    Given a data frame with string columns, a model is trained to predict observed values in label
    column using values observed in other columns.

    The model can then be used to impute missing values.

    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param output_path: path to store model and metrics
    :param num_hash_buckets: number of hash buckets used for the n-gram hashing vectorizer, only
                                used for non-numerical input columns, ignored otherwise
    :param num_labels: number of imputable values considered after, only used for non-numerical
                        input columns, ignored otherwise
    :param tokens: string, 'chars' or 'words' (default 'chars'), determines tokenization strategy
                for n-grams, only used for non-numerical input columns, ignored otherwise
    :param numeric_latent_dim: int, number of latent dimensions for hidden layer of NumericalFeaturizers;
                only used for numerical input columns, ignored otherwise


    Example usage:


    from datawig.simple_imputer import SimpleImputer
    import pandas as pd

    fn_train = os.path.join(datawig_test_path, "resources", "shoes", "train.csv.gz")
    fn_test = os.path.join(datawig_test_path, "resources", "shoes", "test.csv.gz")

    df_train = pd.read_csv(training_data_files)
    df_test = pd.read_csv(testing_data_files)

    output_path = "imputer_model"

    # set up imputer model
    imputer = SimpleImputer( input_columns=['item_name', 'bullet_point'], output_column='brand')

    # train the imputer model
    imputer = imputer.fit(df_train)

    # obtain imputations
    imputations = imputer.predict(df_test)

    """

    def __init__(self,
                 input_columns: List[str],
                 output_column: str,
                 output_path: str = "",
                 num_hash_buckets: int = int(2 ** 15),
                 num_labels: int = 100,
                 tokens: str = 'chars',
                 numeric_latent_dim: int = 100,
                 numeric_hidden_layers: int = 1
                 ) -> None:

        for col in input_columns:
            if not isinstance(col, str):
                raise ValueError("SimpleImputer.input_columns must be str type, was {}".format(type(col)))

        if not isinstance(output_column, str):
            raise ValueError("SimpleImputer.output_column must be str type, was {}".format(type(output_column)))

        self.input_columns = input_columns
        self.output_column = output_column
        self.num_hash_buckets = num_hash_buckets
        self.num_labels = num_labels
        self.tokens = tokens
        self.numeric_latent_dim = numeric_latent_dim
        self.numeric_hidden_layers = numeric_hidden_layers
        self.output_path = output_path
        self.imputer = None
        self.hpo_results = None
        self.numeric_columns = []
        self.string_columns = []

    def check_data_types(self, data_frame: pd.DataFrame) -> None:
        """

        Checks whether a column contains string or numeric data

        :param data_frame:
        :return:
        """
        self.numeric_columns = [c for c in self.input_columns if is_numeric_dtype(data_frame[c])]
        self.string_columns = list(set(self.input_columns) - set(self.numeric_columns))

        logger.info(
            "Assuming {} numeric input columns: {}".format(len(self.numeric_columns),
                                                           ", ".join(self.numeric_columns)))
        logger.info("Assuming {} string input columns: {}".format(len(self.string_columns),
                                                                  ", ".join(self.string_columns)))

    def fit_hpo(self,
                train_df: pd.DataFrame,
                test_df: pd.DataFrame = None,
                ctx: mx.context = get_context(),
                learning_rate: float = 1e-3,
                num_epochs: int = 10,
                patience: int = 3,
                test_split: float = .1,
                weight_decay: List[float] = None,
                batch_size: int = 16,
                num_hash_bucket_candidates: List[float] = None,
                tokens_candidates: List[str] = None,
                numeric_latent_dim_candidates: List[int] = None,
                numeric_hidden_layers_candidates: List[int] = None,
                hpo_max_train_samples: int = 10000,
                normalize_numeric: bool = True,
                final_fc_hidden_units: List[List[int]] = None,
                learning_rate_candidates: List[float] = None) -> Any:

        """

        Fits an imputer model with gridsearch hyperparameter optimization (hpo)

        Grids are specified using the *_candidates arguments.

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param ctx: List of mxnet contexts (if no gpu's available, defaults to [mx.cpu()])
                    User can also pass in a list gpus to be used, ex. [mx.gpu(0), mx.gpu(2), mx.gpu(4)]
        :param learning_rate: learning rate for stochastic gradient descent (default 4e-4)
        :param num_epochs: maximal number of training epochs (default 10)
        :param patience: used for early stopping; after [patience] epochs with no improvement,
                            training is stopped. (default 3)
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        :param weight_decay: regularizer (default 0)
        :param batch_size (default 16)
        :param num_hash_bucket_candidates: candidates for gridsearch hyperparameter
                            optimization (default [2**10, 2**13, 2**15, 2**18, 2**20])
        :param tokens_candidates: candidates for tokenization (default ['words', 'chars'])
        :param numeric_latent_dim_candidates: candidates for latent dimensionality of
                            numerical features (default [10, 50, 100])
        :param numeric_hidden_layers_candidates: candidates for number of hidden layers of
                            numerical features (default [0, 1, 2])
        :param learning_rate_candidates: candidates for learning rate (default [1e-1, 1e-2, 1e-3])
        :param hpo_max_train_samples: training set size for hyperparameter optimization
        :param normalize_numeric: boolean indicating whether or not to normalize numeric values
        :param final_fc_hidden_units: list of lists w/ dimensions for FC layers after the
                            final concatenation (NOTE: for HPO, this expects a list of lists)
        :return: trained SimpleImputer model
        """

        if weight_decay is None:
            weight_decay = [0]

        if num_hash_bucket_candidates is None:
            num_hash_bucket_candidates = [2 ** 10, 2 ** 15, 2 ** 20]

        if tokens_candidates is None:
            tokens_candidates = ['words', 'chars']

        if numeric_latent_dim_candidates is None:
            numeric_latent_dim_candidates = [10, 50, 100]

        if numeric_hidden_layers_candidates is None:
            numeric_hidden_layers_candidates = [0, 1, 2]

        if final_fc_hidden_units is None:
            final_fc_hidden_units = [[100]]

        if learning_rate_candidates is None:
            learning_rate_candidates = [1e-1, 1e-2, 1e-3]

        self.check_data_types(train_df)

        # enable users to define a fixed weight decay
        store_weight_decay = []
        if isinstance(weight_decay, float):
            store_weight_decay.append(weight_decay)
            weight_decay = store_weight_decay

        if len(self.numeric_columns) == 0:
            hps = pd.DataFrame(
                list(itertools.product(num_hash_bucket_candidates, tokens_candidates,
                                    learning_rate_candidates, final_fc_hidden_units)),
                columns=['num_hash_buckets', 'tokens', 'learning_rate', 'final_fc_dim'])

        elif len(self.string_columns) == 0:
            hps = pd.DataFrame(
                list(itertools.product(numeric_latent_dim_candidates, numeric_hidden_layers_candidates,
                                       learning_rate_candidates, final_fc_hidden_units)),
                columns=['numeric_latent_dim', 'numeric_hidden_layers', 'learning_rate', 'final_fc_dim'])
        else:
            hps = pd.DataFrame(
                list(itertools.product(
                    num_hash_bucket_candidates,
                    tokens_candidates,
                    numeric_latent_dim_candidates,
                    numeric_hidden_layers_candidates,
                    learning_rate_candidates,
                    final_fc_hidden_units)),
                columns=['num_hash_buckets', 'tokens', 'numeric_latent_dim', 'numeric_hidden_layers',
                         'learning_rate', 'final_fc_dim'])

        label_column = []

        if is_numeric_dtype(train_df[self.output_column]):
            label_column = [NumericalEncoder(self.output_column, normalize=normalize_numeric)]
            logger.info("Assuming numeric output column: {}".format(self.output_column))
        else:
            label_column = [CategoricalEncoder(self.output_column, max_tokens=self.num_labels)]
            logger.info("Assuming categorical output column: {}".format(self.output_column))

        train_df_hpo, test_df_hpo = random_split(
            train_df.sample(n=min(len(train_df), hpo_max_train_samples)).copy())

        hpo_results = []

        for _, hyper_param in hps.iterrows():

            data_encoders = []
            data_columns = []

            if len(self.string_columns) > 0:
                string_feature_column = "ngram_features-" + rand_string(10)
                data_encoders += [BowEncoder(input_columns=self.string_columns,
                                             output_column=string_feature_column,
                                             max_tokens=hyper_param['num_hash_buckets'],
                                             tokens=hyper_param['tokens'])]
                data_columns += [BowFeaturizer(field_name=string_feature_column,
                                               vocab_size=hyper_param['num_hash_buckets'])]

            if len(self.numeric_columns) > 0:
                numerical_feature_column = "numerical_features-" + rand_string(10)
                data_encoders += [NumericalEncoder(input_columns=self.numeric_columns,
                                                   output_column=numerical_feature_column,
                                                   normalize=normalize_numeric)]
                data_columns += [NumericalFeaturizer(field_name=numerical_feature_column,
                                                     numeric_latent_dim=hyper_param['numeric_latent_dim'],
                                                     numeric_hidden_layers=hyper_param['numeric_hidden_layers'])]

           
            imputer = Imputer(data_encoders=data_encoders,
                                data_featurizers=data_columns,
                                label_encoders=label_column,
                                output_path=self.output_path)

            imputer.fit(train_df_hpo.copy(),
                        None,
                        ctx,
                        hyper_param['learning_rate'],
                        num_epochs,
                        patience,
                        test_split,
                        weight_decay[0],
                        batch_size)

            _, metrics = imputer.transform_and_compute_metrics(test_df_hpo)

            if is_numeric_dtype(train_df[self.output_column]):
                logger.info(
                    "Trained numerical imputer for hps {} on {} samples, MSE: {}".format(
                        hyper_param.to_dict(), len(train_df_hpo),
                        metrics[self.output_column]))
                hyper_param['mse'] = metrics[self.output_column]
            else:
                logger.info(
                    "Trained categorical imputer for hps {} on {} samples, F1: {}".format(
                        hyper_param.to_dict(), len(train_df_hpo),
                        metrics[self.output_column][
                            'avg_f1']))
                hyper_param['f1'] = metrics[self.output_column]['avg_f1']

            hpo_results.append(hyper_param)

        hpo_results_df = pd.DataFrame(hpo_results)

        if is_numeric_dtype(train_df[self.output_column]):
            best_hps = hpo_results_df.sort_values(by='mse', ascending=True).iloc[0].to_dict()
            logger.info("Best hyperparameters for numerical imputation {}".format(best_hps))
        else:
            best_hps = hpo_results_df.sort_values(by='f1', ascending=False).iloc[0].to_dict()
            logger.info("Best hyperparameters for categorical imputation {}".format(best_hps))

        data_encoders, data_columns = [], []

        if len(self.string_columns) > 0:
            data_encoders = [BowEncoder(input_columns=self.string_columns,
                                        output_column=string_feature_column,
                                        max_tokens=best_hps['num_hash_buckets'],
                                        tokens=best_hps['tokens'])]
            data_columns = [BowFeaturizer(field_name=string_feature_column,
                                          vocab_size=best_hps['num_hash_buckets'])]

        if len(self.numeric_columns) > 0:
            data_encoders += [NumericalEncoder(input_columns=self.numeric_columns,
                                               output_column=numerical_feature_column,
                                               normalize=True)]
            data_columns += [NumericalFeaturizer(field_name=numerical_feature_column,
                                                 numeric_latent_dim=best_hps['numeric_latent_dim'],
                                                 numeric_hidden_layers=best_hps['numeric_hidden_layers']
                                                 )]

        self.imputer = Imputer(data_encoders=data_encoders,
                               data_featurizers=data_columns,
                               label_encoders=label_column,
                               output_path=self.output_path)

        logger.info("Retraining on {} samples".format(len(train_df)))

        self.output_path = self.imputer.output_path

        self.imputer = self.imputer.fit(train_df, test_df, ctx, learning_rate, num_epochs,
                                        patience, test_split, weight_decay[0])

        self.hpo_results = hpo_results

        self.save()

        return self

    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            ctx: mx.context = get_context(),
            learning_rate: float = 4e-3,
            num_epochs: int = 10,
            patience: int = 3,
            test_split: float = .1,
            weight_decay: float = 0.,
            batch_size: int = 16,
            final_fc_hidden_units: List[int] = None,
            calibrate: bool = True) -> Any:
        """

        Trains and stores imputer model

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param ctx: List of mxnet contexts (if no gpu's available, defaults to [mx.cpu()])
                    User can also pass in a list gpus to be used, ex. [mx.gpu(0), mx.gpu(2), mx.gpu(4)]
        :param learning_rate: learning rate for stochastic gradient descent (default 4e-4)
        :param num_epochs: maximal number of training epochs (default 10)
        :param patience: used for early stopping; after [patience] epochs with no improvement,
                            training is stopped. (default 3)
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        :param weight_decay: regularizer (default 0)
        :param batch_size (default 16)
        :param final_fc_hidden_units: list dimensions for FC layers after the final concatenation

        """

        if final_fc_hidden_units is None:
            final_fc_hidden_units = [100]

        self.check_data_types(train_df)

        data_encoders = []
        data_columns = []

        if len(self.string_columns) > 0:
            string_feature_column = "ngram_features-" + rand_string(10)
            data_encoders += [BowEncoder(input_columns=self.string_columns,
                                         output_column=string_feature_column,
                                         max_tokens=self.num_hash_buckets,
                                         tokens=self.tokens)]
            data_columns += [
                BowFeaturizer(field_name=string_feature_column, vocab_size=self.num_hash_buckets)]

        if len(self.numeric_columns) > 0:
            numerical_feature_column = "numerical_features-" + rand_string(10)

            data_encoders += [NumericalEncoder(input_columns=self.numeric_columns,
                                               output_column=numerical_feature_column)]

            data_columns += [
                NumericalFeaturizer(field_name=numerical_feature_column, numeric_latent_dim=self.numeric_latent_dim,
                                    numeric_hidden_layers=self.numeric_hidden_layers)]

        label_column = []

        if is_numeric_dtype(train_df[self.output_column]):
            label_column = [NumericalEncoder(self.output_column, normalize=True)]
            logger.info("Assuming numeric output column: {}".format(self.output_column))
        else:
            label_column = [CategoricalEncoder(self.output_column, max_tokens=self.num_labels)]
            logger.info("Assuming categorical output column: {}".format(self.output_column))

        self.imputer = Imputer(data_encoders=data_encoders,
                               data_featurizers=data_columns,
                               label_encoders=label_column,
                               output_path=self.output_path)

        self.output_path = self.imputer.output_path

        self.imputer = self.imputer.fit(train_df, test_df, ctx, learning_rate, num_epochs, patience,
                                        test_split,
                                        weight_decay, batch_size,
                                        final_fc_hidden_units=final_fc_hidden_units,
                                        calibrate=calibrate)
        self.save()

        return self

    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                imputation_suffix: str = "_imputed",
                score_suffix: str = "_imputed_proba"):
        """
        Imputes most likely value if it is above a certain precision threshold determined on the
            validation set
        Precision is calculated as part of the `datawig.evaluate_and_persist_metrics` function.

        Returns original dataframe with imputations and respective likelihoods as estimated by
        imputation model; in additional columns; names of imputation columns are that of the label
        suffixed with `imputation_suffix`, names of respective likelihood columns are suffixed
        with `score_suffix`

        :param data_frame:   data frame (pandas)
        :param precision_threshold: double between 0 and 1 indicating precision threshold
        :param imputation_suffix: suffix for imputation columns
        :param score_suffix: suffix for imputation score columns
        :return: data_frame original dataframe with imputations and likelihood in additional column
        """
        imputations = self.imputer.predict(data_frame, precision_threshold, imputation_suffix,
                                           score_suffix)

        return imputations

    def save(self):
        """

        Saves model to disk; mxnet module and imputer are stored separately

        """
        self.imputer.save()
        simple_imputer_params = {k: v for k, v in self.__dict__.items() if k != 'imputer'}
        pickle.dump(simple_imputer_params,
                    open(os.path.join(self.output_path, "simple_imputer.pickle"), "wb"))

    def load_metrics(self) -> Dict[str, Any]:
        """
        Loads various metrics of the internal imputer model, returned as dictionary
        :return: Dict[str,Any]
        """
        assert os.path.isfile(self.imputer.metrics_path), "Metrics File {} not found".format(
            self.imputer.metrics_path)
        metrics = json.load(open(self.imputer.metrics_path))[self.output_column]
        return metrics

    @staticmethod
    def load(output_path: str) -> Any:
        """

        Loads model from output path

        :param output_path: output_path field of trained SimpleImputer model
        :return: SimpleImputer model

        """

        logger.info("Output path for loading Imputer {}".format(output_path))
        # load pickled model
        simple_imputer_params = pickle.load(
            open(os.path.join(output_path, "simple_imputer.pickle"), "rb"))

        # get constructor args
        constructor_args = inspect.getfullargspec(SimpleImputer.__init__)[0]
        constructor_args = [arg for arg in constructor_args if arg != 'self']

        # get those params that are needed for __init__
        constructor_params = {k: v for k, v in simple_imputer_params.items() if
                              k in constructor_args}
        # instantiate SimpleImputer
        simple_imputer = SimpleImputer(**constructor_params)
        # set all other fields
        for key, value in simple_imputer_params.items():
            if key not in constructor_args:
                setattr(simple_imputer, key, value)
        # set imputer model
        simple_imputer.imputer = Imputer.load(output_path)

        return simple_imputer
