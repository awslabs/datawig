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
- epochs much faster in sklean, how much?

"""



df = pd.read_csv('mae_train_dataset.csv')
# df = df.loc[df['finish'].isin(df['finish'].value_counts().index[:2])]
df_train, df_test = random_split(df, split_ratios=[0.5, 0.5])


si = ScikitImputer(['title', 'text'], 'finish')
si.fit(df_train, df_test)
p = si.predict(df_test)
probas = si.predict_proba(df_test)
# c = calibration_curve(df_test['finish'].values, probas[:, 1])
# e = np.abs(c[0] - c[1]).mean()
aa = 1
########################

# # Initialize a SimpleImputer model
imputer = SimpleImputer(
    input_columns=['title', 'text'],  # columns containing information about the column we want to impute
    output_column='finish',  # the column we'd like to impute values for
    output_path='imputer_model'  # stores model data and metrics
)

# Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=10, calibrate=False)

# Impute missing values and return original dataframe with predictions
predictions = imputer.predict(df_test)
# predictions = imputer.pp(df_test)
# c2 = calibration_curve(df_test['finish'].values, predictions['finish'][:, 1])
# e2 = np.abs(c2[0] - c2[1]).mean()
# Calculate f1 score for true vs predicted values
f1 = f1_score(predictions['finish'], predictions['finish_imputed'], average='weighted')

# Print overall classification report
print(classification_report(df_test['finish'], predictions['finish_imputed']))
print(classification_report(df_test['finish'], p))
b = 1