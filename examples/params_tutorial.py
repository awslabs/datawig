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

import pandas as pd

from datawig import SimpleImputer
from datawig.utils import random_split

import numpy as np


"""
Text Data
"""
df = pd.read_csv('mae_train_dataset.csv').sample(n=1000)
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

# Fit a Model Without HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_text_model',
    num_hash_buckets=2 ** 15,
    tokens='chars'
)

imputer_text.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=5,
    final_fc_hidden_units=[512])

# Fit a Model With HPO
imputer_text = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path='imputer_model',
)

imputer_text.fit_hpo(
    train_df=df_train,
    num_epochs=5,
    num_hash_bucket_candidates=[2 ** 10, 2 ** 15],
    tokens_candidates=['chars', 'words']
)

# ------------------------------------------------------------------------------------

"""
Numerical Data
"""
# Generate synthetic numerical data
n_samples = 100
numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
df = pd.DataFrame({
    'x': numeric_data,
    '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,))
})
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

# Fit a model without HPO
imputer_numeric = SimpleImputer(
    input_columns=['x'],
    output_column="*2",
    output_path='imputer_numeric_model'
)

imputer_numeric.fit(
    train_df=df_train,
    learning_rate=1e-4,
    num_epochs=5
)

# Fit a model with HPO
imputer_numeric = SimpleImputer(
    input_columns=['x'],
    output_column="*2",
    output_path='imputer_numeric_model',
)

imputer_numeric.fit_hpo(
    train_df=df_train,
    num_epochs=5,
    num_hash_bucket_candidates=[2 ** 10, 2 ** 15],
)
