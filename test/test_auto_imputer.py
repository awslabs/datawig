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

DataWig AutoImputer tests

"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, mean_absolute_error

from datawig.utils import (logger, rand_string, random_split, generate_df_numeric,
                           generate_df_string)

from datawig.autogluon_imputer import AutoGluonImputer

warnings.filterwarnings("ignore")

logger.setLevel("INFO")

def test_numeric_or_text_imputer(test_dir, data_frame):
    """
    Tests AutoGluonImputer with default options

    """
    output_path = ''

    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 30
    vocab_size = int(2 ** 10)

    # generate some random data
    random_data = data_frame(feature_col=feature_col,
                             label_col=label_col,
                             vocab_size=vocab_size,
                             num_labels=num_labels,
                             num_words=seq_len,
                             n_samples=n_samples)

    numeric_data = np.random.uniform(-np.pi, np.pi, (n_samples,))
    df = pd.DataFrame({
        'x': numeric_data,
        '*2': numeric_data * 2. + np.random.normal(0, .1, (n_samples,)),
        '**2': numeric_data ** 2 + np.random.normal(0, .1, (n_samples,)),
        feature_col: random_data[feature_col].values,
        label_col: random_data[label_col].values
    })

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(test_dir, "tmp", "real_data_experiment_numeric")

    imputer_numeric_linear = AutoGluonImputer(
        input_columns=['x', feature_col],
        output_column="*2"
    ).fit(
        train_df=df_train
    )

    imputer_numeric_linear.predict(df_test, inplace=True)

    assert mean_absolute_error(df_test['*2'], df_test['*2_imputed']) < 1.0

    imputer_numeric = AutoGluonImputer(
        input_columns=['x', feature_col],
        output_column="**2"
    ).fit(
        train_df=df_train
    )


    imputer_numeric.predict(df_test, inplace=True)

    assert mean_absolute_error(df_test['**2'], df_test['**2_imputed']) < 0.1

    imputer_string = AutoGluonImputer(
        input_columns=[feature_col, 'x'],
        output_column=label_col,
        output_path=output_path
    ).fit(
        train_df=df_train
    )

    imputer_string.predict(df_test, inplace=True)

    assert f1_score(df_test[label_col], df_test[label_col + '_imputed'].fillna(''), average="weighted") > .7


def test_imputer_complete():
    """
    Tests the complete functionality of SimpleImputer
    :param data_frame:
    """

    feature_col = "string_feature"
    label_col = "label"
    feature_col_numeric = "numeric_feature"
    label_col_numeric = "numeric_label"

    num_samples = 1000
    num_labels = 3
    seq_len = 20

    missing_ratio = .2

    df_string = generate_df_string(
    num_labels=num_labels,
    num_words=seq_len,
    num_samples=num_samples,
    label_column_name=label_col,
    data_column_name=feature_col)

    df_numeric = generate_df_numeric(num_samples=num_samples,
                                 label_column_name=label_col_numeric,
                                 data_column_name=feature_col_numeric)

    df = pd.concat([
    df_string[[feature_col, label_col]],
    df_numeric[[feature_col_numeric, label_col_numeric]]
    ], ignore_index=True, axis=1)
    df.columns = [feature_col, label_col, feature_col_numeric, label_col_numeric]

    df_orig = df.copy(deep=True)
    df_missing = df.copy(deep=True)

    # delete some entries
    missing = (np.random.random(len(df) * len(df.columns)) < missing_ratio) \
                .reshape((len(df),len(df.columns)))
    df_missing[missing] = np.nan

    feature_col_missing = df[feature_col].isnull()
    label_col_missing = df[label_col].isnull()


    df_imputed = AutoGluonImputer.complete(data_frame=df_missing)

    f1 = f1_score(
            df_orig.loc[df_missing[label_col].isna(),label_col], 
            df_imputed.loc[df_missing[label_col].isna(),label_col].fillna(''), average='weighted')

    mae = mean_absolute_error(
            df_orig.loc[df_missing[label_col_numeric].isna(),label_col_numeric], 
            df_imputed.loc[df_missing[label_col_numeric].isna(),label_col_numeric].fillna(0))

    assert f1 < .9
    assert mae < 1.

