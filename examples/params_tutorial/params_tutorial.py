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

from datawig import SimpleImputer
from datawig.utils import random_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd
import os

'''
Text Data
'''
df = pd.read_csv('../finish_val_data_sample.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#Fit a Model Without HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_text_model',
    num_hash_buckets=2**15,
    tokens='chars'
    )

imputer_text.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=50)

#Fit a Model With HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_model',
    )

imputer_text.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    num_hash_bucket_candidates=[2**10, 2**15],
    tokens_candidates=['chars', 'words']
    )

#------------------------------------------------------------------------------------

'''
Numerical Data
'''
#Generate synthetic numerical data
numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
df = pd.DataFrame({
    'x': numeric_data,
    '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,)),
    '**2': numeric_data ** 2 + np.random.normal(0, .1, (n_samples,)),
    feature_col: random_data[feature_col].values,
    label_col: random_data[label_col].values
    })
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#Fit a model without HPO
imputer_numeric = SimpleImputer(
    input_columns=['x', feature_col],
    output_column="*2",
    output_path='imputer_numeric_model',
    latent_dim = 512,
    hidden_layers = 1
    )

imputer_numeric.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=50
    )

#Fit a model with HPO
imputer_numeric = SimpleImputer(
    input_columns=['x', feature_col],
    output_column="*2",
    output_path=output_path,
    )

imputer_numeric.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    latent_dim_candidates=[50, 100],
    hidden_layers_candidates=[0, 2]
    )