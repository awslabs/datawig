# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from abc import abstractmethod
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import joblib


class SimpleImputerBackend:
    def __init__(self, input_columns, output_column):
        self.input_columns = input_columns
        self.output_column = output_column

    def data_types_for(self, data_frame: pd.DataFrame) -> dict:
        result = dict()
        for col in self.input_columns:
            if is_numeric_dtype(data_frame[col]):
                result[col] = 'numeric'
            else:
                result[col] = 'string'

        output_column_type = 'numeric' if is_numeric_dtype(data_frame[self.output_column]) else 'string'
        result['output_type'] = output_column_type

        return result

    @abstractmethod
    def calibrate(self, test_df):
        pass

    @abstractmethod
    def fit(self, train_df, test_df):
        pass

    @abstractmethod
    def explain(self):
        pass

    @abstractmethod
    def explain_instance(self):
        pass

    @abstractmethod
    def predict(self, df):
        pass

    @abstractmethod
    def fit_hpo(self, grid):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


# class MXNetImputer(SimpleImputerBackend):
#     def __init__(self, input_columns, output_column):
#         super().__init__(input_columns, output_column)
#         self.model = None
#
#     def fit(self, train_df, test_df):
#         pass


class ScikitImputer(SimpleImputerBackend):
    def __init__(self, input_columns, output_column):
        super().__init__(input_columns, output_column)
        self.model = None
        self.__class_prototypes = None

    def fit(self, train_df, test_df, calibrate: bool = True):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import SGDClassifier, SGDRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.preprocessing import StandardScaler

        data_types = self.data_types_for(train_df)

        data_type_encoders = {
            'string': TfidfVectorizer(),
            'numeric': StandardScaler()
        }

        transformers = ColumnTransformer([(col, data_type_encoders[data_types[col]], col) for col in self.input_columns])

        estimator = SGDClassifier('log', max_iter=100) if data_types['output_type'] == 'string' else SGDRegressor('squared_loss')
        if calibrate:
            estimator = CalibratedClassifierCV(estimator, cv=5)

        clf = Pipeline([('transformers', transformers), ('estimator', estimator)])

        self.model = clf

        self.model.fit(train_df[self.input_columns], train_df[self.output_column])

        # --- class prototypes
        # train probas
        Y = StandardScaler().fit_transform(self.model.predict_proba(train_df))
        # features
        Xs = [StandardScaler(with_mean=False).fit_transform(encoder.transform(train_df[col])) for col, encoder in self.model.named_steps['transformers'].named_transformers_.items()]
        As = [X.T.dot(Y) for X in Xs]
        self.__class_prototypes = As

        return self.model

    def predict(self, df: pd.DataFrame) -> np.array:
        return self.model.predict(df)

    def explain(self, label: str, k: int = 10) -> dict:
        pattern_index = (self.model.classes_ == label).argmax()

        As = [A[:, pattern_index] for A in self.__class_prototypes]

        col_explanations = {}
        for idx, (col, encoder) in enumerate(self.model.named_steps['transformers'].named_transformers_.items()):
            i = encoder.inverse_transform(As[idx])[0]
            tokens = i[As[idx].argsort()[::-1][:k]].tolist()
            col_explanations[col] = tokens

        return col_explanations

    def explain_instance(self, instance: pd.Series, k: int = 10) -> dict():
        instance = pd.DataFrame([instance])
        label = self.model.predict(instance)

        pattern_index = (self.model.classes_ == label).argmax()

        As = [A[:, pattern_index] for A in self.__class_prototypes]

        col_explanations = {}
        for idx, (col, encoder) in enumerate(self.model.named_steps['transformers'].named_transformers_.items()):
            x = encoder.transform(instance[col])
            instance_patterns = np.multiply(x, As[idx])[0]
            i = encoder.inverse_transform(As[idx])[0]
            tokens = i[instance_patterns.toarray()[0].argsort()[::-1][:k]].tolist()
            col_explanations[col] = tokens

        return col_explanations

    def save(self, output_path: str):
        joblib.dump(self, output_path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)

    def predict_proba(self, df: pd.DataFrame):
        return self.model.predict_proba(df)
