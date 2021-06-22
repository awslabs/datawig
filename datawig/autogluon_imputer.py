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

AutoGluon Imputer:
Imputes missing values in tables based on autogluon-tabular

"""
import os, shutil, warnings

from typing import List, Dict, Any, Callable

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import precision_recall_curve, classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split


class TargetColumnException(Exception):
    """Raised when a target column cannot be used as label for a supervised learning model"""
    pass

from autogluon.tabular import TabularPredictor

class AutoGluonImputer():

    """

    AutoGluonImputer model

    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param precision_threshold: precision threshold for imputation of categorical values; if predictions on a validation set were below that threshold, no imputation will be made
    :param numerical_confidence_quantile: confidence quantile for imputation of numerical values (very experimental)
    :param verbosity: verbosity level

    Example usage:


    """

    def __init__(self,
             output_column: str = None,
             input_columns: List[str] = None,
             precision_threshold: float = 0.0,
             numerical_confidence_quantile: float = 0.0,
             verbosity: int = 0) -> None:

        self.input_columns = input_columns
        self.output_column = output_column
        self.precision_threshold = precision_threshold
        self.numerical_confidence_quantile = numerical_confidence_quantile
        self.verbosity = verbosity
        self.predictor = None


    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            test_split: float = .1,
            time_limit: int = 30) -> Any:

        """

        Trains AutoGluonImputer model for a single column

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        :param time_limit: time limit for AutoGluon in seconds
        """
        if not test_df:
            train_df,  test_df  =  train_test_split(train_df.copy(), test_size=test_split)

        if not self.input_columns or len(self.input_columns) == 0:
            self.input_columns = [c for c in train_df.columns if c is not self.output_column]

        if not is_numeric_dtype(train_df[self.output_column]):
            if train_df[self.output_column].value_counts().max() < 10:
                raise TargetColumnException("Maximum class count below 10, cannot train imputation model")

            self.predictor = TabularPredictor(label=self.output_column,
                                              problem_type='multiclass',
                                              verbosity=0).\
                                                 fit(train_data=train_df.dropna(subset=[self.output_column]),
                                                        time_limit=time_limit,
                                                        verbosity=self.verbosity)
            y_test = test_df.dropna(subset=[self.output_column])

            # prec-rec curves for finding the likelihood thresholds for minimal precision
            self.precision_thresholds = {}
            probas = self.predictor.predict_proba(y_test)

            for col_name in probas.columns:
                prec, rec, thresholds = precision_recall_curve(y_test[self.output_column]==col_name,
                                                              probas[col_name], pos_label=True)
                threshold_idx = max(min((prec >= self.precision_threshold).nonzero()[0][0], len(thresholds)-1), 0)
                threshold_for_minimal_precision = thresholds[threshold_idx]
                self.precision_thresholds[col_name] = threshold_for_minimal_precision

            self.classification_metrics = classification_report(y_test[self.output_column],
                                                                self.predictor.predict(y_test))

        else:

            if self.numerical_confidence_quantile == 0.:
                quantile = 1e-5
            else:
                quantile = self.numerical_confidence_quantile

            self.quantiles = [quantile, .5, 1-quantile]

            self.predictor = TabularPredictor(
                                            label=self.output_column,
                                            quantile_levels=self.quantiles,
                                            problem_type='quantile',
                                            verbosity=0)\
                                                .fit(train_data=train_df.dropna(subset=[self.output_column]),
                                                    time_limit=time_limit)

            y_test = test_df[self.output_column].dropna()
            y_pred = self.predictor.predict(test_df.dropna(subset=[self.output_column]))
            self.predictor.mean_absolute_error = mean_absolute_error(y_test, y_pred[self.quantiles[1]])

        return self

    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                numerical_confidence_interval: float = 1.0,
                imputation_suffix: str = "_imputed",
                inplace: bool = False):
        """
        Imputes most likely value if it is above a certain precision threshold determined on the
            validation set

        Returns original dataframe with imputations and respective likelihoods as estimated by
        imputation model; in additional columns; names of imputation columns are that of the label
        suffixed with `imputation_suffix`, names of respective likelihood columns are suffixed
        with `score_suffix`

        :param data_frame:   data frame (pandas)
        :param precision_threshold: double between 0 and 1 indicating precision threshold categorical imputation
        :param numerical_confidence_interval: double between 0 and 1 indicating confidence quantile for numerical imputation
        :param imputation_suffix: suffix for imputation columns
        :param inplace: add column with imputed values and column with confidence scores to data_frame, returns the
            modified object (True). Create copy of data_frame with additional columns, leave input unmodified (False).
        :return: data_frame original dataframe with imputations and likelihood in additional column
        """
        if not inplace:
            df = data_frame.copy(deep=True)
        else:
            df = data_frame

        if not is_numeric_dtype(df[self.output_column]):
            imputations = self.predictor.predict(df)
            probas = self.predictor.predict_proba(df)
            for label in self.precision_thresholds.keys():
                above_precision = (imputations == label)
                if self.precision_threshold > 0:
                    above_precision = above_precision & \
                        (probas[label] >= self.precision_thresholds[label])
                df.loc[above_precision, self.output_column + "_imputed"] = label
        else:
            imputations = self.predictor.predict(df)
            if self.numerical_confidence_quantile > 0:
                confidence_tube = imputations[self.quantiles[2]] - imputations[self.quantiles[0]]
                error_smaller_than_confidence_tube = confidence_tube > self.predictor.mean_absolute_error
                df.loc[error_smaller_than_confidence_tube, self.output_column + "_imputed"] = \
                    imputations.loc[error_smaller_than_confidence_tube, self.quantiles[1]]

        return df

    @staticmethod
    def complete(data_frame: pd.DataFrame,
                 precision_threshold: float = 0.0,
                 numeric_confidence_quantile = 0.0,
                 inplace: bool = False,
                 time_limit: float = 60.,
                 verbosity=0):
        """
        Given a dataframe with missing values, this function detects all imputable columns, trains an imputation model
        on all other columns and imputes values for each missing value using AutoGluon.

        :param data_frame: original dataframe
        :param precision_threshold: precision threshold for categorical imputations (default: 0.0)
        :param inplace: whether or not to perform imputations inplace (default: False)
        :param verbose: verbosity level, values > 0 log to stdout (default: 0)
        :param output_path: path to store model and metrics
        :return: dataframe with imputations
        """

        # TODO: should we expose temporary dir for model serialization to avoid crashes due to not-writable dirs?

        missing_mask = data_frame.copy().isnull()

        if inplace is False:
            data_frame = data_frame.copy()

        for output_col in data_frame.columns:

            input_cols = list(set(data_frame.columns) - set([output_col]))

            # train on all observed values
            idx_missing = missing_mask[output_col]
            try:
                imputer = AutoGluonImputer(input_columns=input_cols,
                                        output_column=output_col,
                                        precision_threshold = 0.0,
                                        numerical_confidence_quantile = 0.05,
                                        verbosity=verbosity)\
                                            .fit(data_frame, time_limit=time_limit)
                tmp = imputer.predict(data_frame)
                data_frame.loc[idx_missing, output_col] = tmp[output_col + "_imputed"]
            except TargetColumnException:
                warnings.warn(f'Could not train model on column {output_col}')
        return data_frame


        def save(self):
            """

            Saves model to disk; mxnet module and imputer are stored separately

            """
            raise(NotImplementedError)

        @staticmethod
        def load(output_path: str) -> Any:
            """

            Loads model from output path

            :param output_path: output_path field of trained SimpleImputer model
            :return: AutoGluonImputer model

            """

            raise(NotImplementedError)
