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


class AutoGluonImputer:

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
