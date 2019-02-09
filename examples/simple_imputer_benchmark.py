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

"""
Load Data
"""
df = pd.read_csv('mae_train_dataset.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])


si = ScikitImputer(['title', 'text'], 'finish')
si.fit(df_train)
p = si.predict(df_test)
aa = 1
########################

# # Initialize a SimpleImputer model
imputer = SimpleImputer(
    input_columns=['title', 'text'],  # columns containing information about the column we want to impute
    output_column='finish',  # the column we'd like to impute values for
    output_path='imputer_model'  # stores model data and metrics
)

# Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=5)

# Impute missing values and return original dataframe with predictions
predictions = imputer.predict(df_test)

# Calculate f1 score for true vs predicted values
f1 = f1_score(predictions['finish'], predictions['finish_imputed'], average='weighted')

# Print overall classification report
print(classification_report(predictions['finish'], predictions['finish_imputed']))
b = 1