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

from pandas.api.types import is_numeric_dtype
from sklearn.metrics import precision_recall_curve, classification_report, mean_absolute_error


class AutoGluonImputer:

    """

    AutoGluonImputer model 

    :param input_columns: list of input column names (as strings)
    :param output_column: output column name (as string)
    :param output_path: path to store model and metrics
   

    Example usage:


    """

    @staticmethod
    def _is_categorical(col: pd.Series,
                        n_samples: int = 100,
                        max_unique_fraction=0.05) -> bool:
        """

        A heuristic to check whether a column is categorical:
        a column is considered categorical (as opposed to a plain text column)
        if the relative cardinality is max_unique_fraction or less.

        :param col: pandas Series containing strings
        :param n_samples: number of samples used for heuristic (default: 100)
        :param max_unique_fraction: maximum relative cardinality.

        :return: True if the column is categorical according to the heuristic

        """

        sample = col.sample(n=n_samples, replace=len(col) < n_samples).unique()

        return sample.shape[0] / n_samples < max_unique_fraction

    def __init__(self,
                 input_columns: List[str],
                 output_column: str,
                 precision_threshold: float = 0.0,
                 numerical_confidence_quantile = 0.0,
                 output_path: str = "") -> None:

        self.input_columns = input_columns
        self.output_column = output_column
        self.precision_threshold = precision_threshold
        self.numerical_confidence_quantile = numerical_confidence_quantile
        self.output_path = output_path
        

    def fit(self,
            train_df: pd.DataFrame,
            test_df: pd.DataFrame = None,
            test_split: float = .1) -> Any:
        """

        Trains and stores imputer model

        :param train_df: training data as dataframe
        :param test_df: test data as dataframe; if not provided, a ratio of test_split of the
                            training data are used as test data
        :param test_split: if no test_df is provided this is the ratio of test data to be held
                            separate for determining model convergence
        """

        if self._is_categorical(train_df[self.output_column]):
        
            self.predictor = TabularPredictor( 
                                                label=self.output_column, 
                                                problem_type='multiclass', 
                                                verbosity=0).\
                                                    fit(train_data=train_df.dropna(subset=[self.output_column]))
            y_test = test_df.dropna(subset=[self.output_column]).drop([self.output_column],axis=1)
           
            # prec-rec curves for finding the likelihood thresholds for minimal precision
            self.precision_thresholds = {}
            probas = self.predictor.predict_proba(y_test)

            for this_label_proba in probas.columns:
                prec, rec, threshold = precision_recall_curve(y_test==this_label_proba.name, proba, pos_label=True)
                threshold_for_minimal_precision = threshold[(prec >= self.categorical_precision_threshold).nonzero()[0][0]]
                self.precision_thresholds[this_label_proba.name] = threshold_for_minimal_precision
            
            self.classification_metrics = classification_report(y_test, self.predictor.predict(y_test))

        else:
            self.quantiles = [ 
                            self.numerical_confidence_quantile,
                            .5,   
                            1-self.numerical_confidence_quantile
                        ]
            self.predictors[col] = TabularPredictor(
                                            label=col, 
                                            quantile_levels=quantiles, 
                                            problem_type='quantile', 
                                            verbosity=0)\
                                                .fit(train_data=train_df.dropna(subset=[self.output_column]))

            y_test = test_df[self.output_column].dropna()
            y_pred = self.predictor.predict(y_test)
            self.predictor.mean_absolute_error = mean_absolute_error(y_test, y_pred[self.quantiles[1]])

    def predict(self,
                data_frame: pd.DataFrame,
                precision_threshold: float = 0.0,
                numerical_confidence_interval: float = 1.0,
                imputation_suffix: str = "_imputed",
                inplace: bool = False):
        """
        Imputes most likely value if it is above a certain precision threshold determined on the
            validation set
        Precision is calculated as part of the `datawig.evaluate_and_persist_metrics` function.

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
        imputations = self.imputer.predict(data_frame, precision_threshold, imputation_suffix,
                                           score_suffix, inplace=inplace)

        return imputations

    @staticmethod
    def complete(data_frame: pd.DataFrame,
                 precision_threshold: float = 0.0,
                 numeric_confidence_quantile = 0.0,
                 inplace: bool = False,
                 output_path: str = "."):
        """
        Given a dataframe with missing values, this function detects all imputable columns, trains an imputation model
        on all other columns and imputes values for each missing value using AutoGluon.

        :param data_frame: original dataframe
        :param precision_threshold: precision threshold for categorical imputations (default: 0.0)
        :param inplace: whether or not to perform imputations inplace (default: False)
        :param hpo: whether or not to perform hyperparameter optimization (default: False)
        :param verbose: verbosity level, values > 0 log to stdout (default: 0)
        :param num_epochs: number of epochs for each imputation model training (default: 100)
        :param iterations: number of iterations for iterative imputation (default: 1)
        :param output_path: path to store model and metrics
        :return: dataframe with imputations
        """

        # TODO: should we expose temporary dir for model serialization to avoid crashes due to not-writable dirs?

        missing_mask = data_frame.copy().isnull()

        if inplace is False:
            data_frame = data_frame.copy()

        numeric_columns = [c for c in data_frame.columns if is_numeric_dtype(data_frame[c])]
        string_columns = list(set(data_frame.columns) - set(numeric_columns))
        logger.debug("Assuming numerical columns: {}".format(", ".join(numeric_columns)))

        col_set = set(numeric_columns + string_columns)

        categorical_columns = [col for col in string_columns if SimpleImputer._is_categorical(data_frame[col])]
        logger.debug("Assuming categorical columns: {}".format(", ".join(categorical_columns)))
        for _ in range(iterations):
            for output_col in set(numeric_columns) | set(categorical_columns):
                # train on all input columns but the to-be-imputed one
                input_cols = list(col_set - set([output_col]))

                # train on all observed values
                idx_missing = missing_mask[output_col]

                imputer = SimpleImputer(input_columns=input_cols,
                                        output_column=output_col,
                                        output_path=os.path.join(output_path, output_col))
                if hpo:
                    imputer.fit_hpo(data_frame.loc[~idx_missing, :],
                                    patience=5 if output_col in categorical_columns else 20,
                                    num_epochs=100,
                                    final_fc_hidden_units=[[0], [10], [50], [100]])
                else:
                    imputer.fit(data_frame.loc[~idx_missing, :],
                                patience=5 if output_col in categorical_columns else 20,
                                num_epochs=num_epochs,
                                calibrate=False)

                tmp = imputer.predict(data_frame, precision_threshold=precision_threshold)
                data_frame.loc[idx_missing, output_col] = tmp[output_col + "_imputed"]

                # remove the directory with logfiles for this column
                shutil.rmtree(os.path.join(output_path, output_col))


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
