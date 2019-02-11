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

from datawig.simple_imputer import SimpleImputer
from datawig.backend import ScikitImputer
from datawig.utils import random_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
from sklearn.calibration import calibration_curve
import numpy as np


"""
- mxnet calibr. seems to work
- encoding takes similar time

"""

df = pd.read_csv('mae_train_dataset.csv')
# df = df.loc[df['finish'].isin(df['finish'].value_counts().index[:2])]
df.loc[:, 'n'] = np.random.randn(df.shape[0])
df.loc[df.finish == 'black', 'n'] += 5
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])
print(df_train.shape)

si = ScikitImputer(['text', 'n'], 'finish')
si.fit(df_train, df_test)
grid = {
    # numeric encoder
    'transformers__n__scaler__with_mean': [True],
    # string encoder
    'transformers__text__max_features': [2**10, 2**12]
}
si.fit_hpo(df_train, grid)
p = si.predict(df_test)
# probas = si.predict_proba(df_test)
# si.save("/tmp/a.jl")
# si2 = ScikitImputer.load("/tmp/a.jl")
# si.explain('black')
# si.explain_instance(df.iloc[0])
# c = calibration_curve(df_test['finish'].values, probas[:, 1])
# e = np.abs(c[0] - c[1]).mean()
print(si.model.best_params_)
print(classification_report(df_test['finish'], p))
aa = 1
########################
#
# # # # Initialize a SimpleImputer model
# imputer = SimpleImputer(
#     # input_columns=['title', 'text'],  # columns containing information about the column we want to impute
#     input_columns=['text'],  # columns containing information about the column we want to impute
#     output_column='finish',  # the column we'd like to impute values for
#     output_path='imputer_model',  # stores model data and metrics
#     is_explainable=True
# )
#
# # Fit an imputer model on the train data
# imputer.fit(train_df=df_train, test_df=df_test, num_epochs=10, calibrate=True)
#
# # # imputer.explain('black')
# # # Impute missing values and return original dataframe with predictions
# predictions = imputer.predict(df_test)
# # predictions = imputer.pp(df_test)
# # c2 = calibration_curve(df_test['finish'].values, predictions['finish'][:, 1])
# # e2 = np.abs(c2[0] - c2[1]).mean()
# # Calculate f1 score for true vs predicted values
# # f1 = f1_score(predictions['finish'], predictions['finish_imputed'], average='weighted')
#
# # Print overall classification report
# # print(classification_report(df_test['finish'], predictions['finish_imputed']))
