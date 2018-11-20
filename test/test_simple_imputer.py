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

DataWig SimpleImputer tests

"""

import os
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score, mean_squared_error

from datawig.column_encoders import BowEncoder
from datawig.mxnet_input_symbols import BowFeaturizer
from datawig.simple_imputer import SimpleImputer
from datawig.utils import logger, rand_string, random_split, generate_df_numeric, generate_df_string

warnings.filterwarnings("ignore")

logger.setLevel("INFO")


def test_simple_imputer_no_string_column_name():
    with pytest.raises(ValueError):
        SimpleImputer([0], '1')
    with pytest.raises(ValueError):
        SimpleImputer(['0'], 1)


def test_simple_imputer_real_data_default_args(test_dir, data_frame):
    """
    Tests SimpleImputer with default options

    """
    feature_col = "string_feature"
    label_col = "label"

    n_samples = 2000
    num_labels = 3
    seq_len = 100
    vocab_size = int(2 ** 15)

    # generate some random data
    random_data = data_frame(feature_col=feature_col,
                             label_col=label_col,
                             vocab_size=vocab_size,
                             num_labels=num_labels,
                             num_words=seq_len,
                             n_samples=n_samples)

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    output_path = os.path.join(test_dir, "tmp", "real_data_experiment_simple")

    df_train_cols_before = df_train.columns.tolist()

    input_columns = [feature_col]

    imputer = SimpleImputer(
        input_columns=input_columns,
        output_column=label_col,
        output_path=output_path
    ).fit(
        train_df=df_train
    )

    logfile = os.path.join(imputer.output_path, 'imputer.log')
    assert os.path.exists(logfile)
    assert os.path.getsize(logfile) > 0

    assert imputer.output_path == output_path
    assert imputer.imputer.data_featurizers[0].__class__ == BowFeaturizer
    assert imputer.imputer.data_encoders[0].__class__ == BowEncoder
    assert set(imputer.imputer.data_encoders[0].input_columns) == set(input_columns)
    assert set(imputer.imputer.label_encoders[0].input_columns) == set([label_col])


    assert all([after == before for after, before in zip(df_train.columns, df_train_cols_before)])

    df_no_label_column = df_test.copy()
    true_labels = df_test[label_col]
    del (df_no_label_column[label_col])
    df_test_cols_before = df_no_label_column.columns.tolist()

    df_test_imputed = imputer.predict(df_no_label_column, inplace=True)

    assert all(
        [after == before for after, before in zip(df_no_label_column.columns, df_test_cols_before)])

    imputed_columns = df_test_cols_before + [label_col + "_imputed", label_col + "_imputed_proba"]

    assert all([after == before for after, before in zip(df_test_imputed, imputed_columns)])

    f1 = f1_score(true_labels, df_test_imputed[label_col + '_imputed'], average="weighted")

    assert f1 > .9

    new_path = imputer.output_path + "-" + rand_string()

    os.rename(imputer.output_path, new_path)

    deserialized = SimpleImputer.load(new_path)
    df_test = deserialized.predict(df_test, imputation_suffix="_deserialized_imputed")
    f1 = f1_score(df_test[label_col], df_test[label_col + '_deserialized_imputed'],
                  average="weighted")

    assert f1 > .9

    retrained_simple_imputer = deserialized.fit(df_train, df_train)

    df_train_imputed = retrained_simple_imputer.predict(df_train.copy(), inplace=True)
    f1 = f1_score(df_train[label_col], df_train_imputed[label_col + '_imputed'], average="weighted")

    assert f1 > .9

    metrics = retrained_simple_imputer.load_metrics()

    assert f1 == metrics['weighted_f1']


def test_numeric_or_text_imputer(test_dir, data_frame):
    """
    Tests SimpleImputer with default options

    """

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

    imputer_numeric_linear = SimpleImputer(
        input_columns=['x', feature_col],
        output_column="*2",
        output_path=output_path
    ).fit(
        train_df=df_train,
        learning_rate=1e-3,
    )

    imputer_numeric_linear.predict(df_test, inplace=True)

    assert mean_squared_error(df_test['*2'], df_test['*2_imputed']) < 1.0

    imputer_numeric = SimpleImputer(
        input_columns=['x', feature_col],
        output_column="**2",
        output_path=output_path
    ).fit(
        train_df=df_train,
        learning_rate=1e-3
    )

    imputer_numeric.predict(df_test, inplace=True)

    assert mean_squared_error(df_test['**2'], df_test['**2_imputed']) < 1.0

    imputer_string = SimpleImputer(
        input_columns=[feature_col, 'x'],
        output_column=label_col,
        output_path=output_path
    ).fit(
        train_df=df_train
    )

    imputer_string.predict(df_test, inplace=True)

    assert f1_score(df_test[label_col], df_test[label_col + '_imputed'], average="weighted") > .7


def test_imputer_hpo_numeric(test_dir):
    """

    Tests SimpleImputer HPO for numeric data/imputation

    """

    N = 200
    numeric_data = np.random.uniform(-np.pi, np.pi, (N,))
    df = pd.DataFrame({
        'x': numeric_data,
        '**2': numeric_data ** 2 + np.random.normal(0, .1, (N,)),
    })

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(test_dir, "tmp", "experiment_numeric_hpo")

    imputer_numeric = SimpleImputer(
        input_columns=['x'],
        output_column="**2",
        output_path=output_path)

    feature_col = 'x'

    hps = {}
    hps[feature_col] = {}
    hps[feature_col]['type'] = ['numeric']
    hps[feature_col]['numeric_latent_dim'] = [30]
    hps[feature_col]['numeric_hidden_layers'] = [1]

    hps['global'] = {}
    hps['global']['final_fc_hidden_units'] = [[]]
    hps['global']['learning_rate'] = [1e-3, 1e-4]
    hps['global']['weight_decay'] = [0]
    hps['global']['num_epochs'] = [200]
    hps['global']['patience'] = [100]
    hps['global']['concat_columns'] = [False]

    imputer_numeric.fit_hpo(df_train, hps=hps)
    results = imputer_numeric.hpo.results

    assert results[results['mse'] == min(results['mse'])]['mse'].iloc[0] < .3


def test_imputer_hpo_text(test_dir, data_frame):
    """

    Tests SimpleImputer HPO with text data and categorical imputations

    """
    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 20

    # generate some random data
    df = data_frame(feature_col=feature_col,
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    df_train, df_test = random_split(df, [.8, .2])

    output_path = os.path.join(test_dir, "tmp", "experiment_text_hpo")

    imputer_string = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=output_path
    )

    hps = {}
    hps[feature_col] = {}
    hps[feature_col]['type'] = ['string']
    hps[feature_col]['tokens'] = [['words'], ['chars']]

    hps['global'] = {}
    hps['global']['final_fc_hidden_units'] = [[]]
    hps['global']['learning_rate'] = [1e-3]
    hps['global']['weight_decay'] = [0]
    hps['global']['num_epochs'] = [30]

    imputer_string.fit_hpo(df_train, hps=hps)

    assert max(imputer_string.hpo.results['f1_micro']) > .9


def test_hpo_all_input_types(test_dir, data_frame):
    """

    Using sklearn advantages: parallelism, distributions of parameters, multiple cross-validation

    """
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 12

    # generate some random data
    df = data_frame(feature_col="string_feature",
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    # add categorical feature
    df['categorical_feature'] = ['foo' if r > .5 else 'bar' for r in np.random.rand(n_samples)]

    # add numerical feature
    df['numeric_feature'] = np.random.rand(n_samples)

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(test_dir, "tmp", "real_data_experiment_text_hpo")

    imputer = SimpleImputer(
        input_columns=['string_feature', 'categorical_feature', 'numeric_feature'],
        output_column='label',
        output_path=output_path
    )

    # Define default hyperparameter choices for each column type (string, categorical, numeric)
    hps = dict()
    hps['global'] = {}
    hps['global']['learning_rate'] = [3e-4]
    hps['global']['weight_decay'] = [1e-8]
    hps['global']['num_epochs'] = [5, 50]
    hps['global']['patience'] = [5]
    hps['global']['batch_size'] = [16]
    hps['global']['final_fc_hidden_units'] = [[]]
    hps['global']['concat_columns'] = [True, False]

    hps['string_feature'] = {}
    hps['string_feature']['max_tokens'] = [2 ** 15]
    hps['string_feature']['tokens'] = [['words', 'chars']]
    hps['string_feature']['ngram_range'] = {}
    hps['string_feature']['ngram_range']['words'] = [(1, 4), (2, 5)]
    hps['string_feature']['ngram_range']['chars'] = [(2, 4), (3, 5)]

    hps['categorical_feature'] = {}
    hps['categorical_feature']['type'] = ['categorical']
    hps['categorical_feature']['max_tokens'] = [2 ** 15]
    hps['categorical_feature']['embed_dim'] = [10]

    hps['numeric_feature'] = {}
    hps['numeric_feature']['normalize'] = [True]
    hps['numeric_feature']['numeric_latent_dim'] = [10]
    hps['numeric_feature']['numeric_hidden_layers'] = [1]

    # user defined score function for hyperparameters
    def calibration_check(true, predicted, confidence):
        """
        expect kwargs: true, predicted, confidence
        here we compute a calibration sanity check
        """
        return (np.mean(true[confidence > .9] == predicted[confidence > .9]),
                np.mean(true[confidence > .5] == predicted[confidence > .5]))

    def coverage_check(true, predicted, confidence):
        return np.mean(confidence > .9)

    uds = [(calibration_check, 'calibration check'),
           (coverage_check, 'coverage at 90')]

    imputer.fit_hpo(df_train,
                    hps=hps,
                    user_defined_scores=uds,
                    num_evals=5,
                    hpo_run_name='test1_')

    imputer.fit_hpo(df_train,
                    hps=hps,
                    user_defined_scores=uds,
                    num_evals=5,
                    hpo_run_name='test2_',
                    max_running_hours=1/3600)

    results = imputer.hpo.results

    assert results[results['global:num_epochs'] == 50]['f1_micro'].iloc[0] > \
           results[results['global:num_epochs'] == 5]['f1_micro'].iloc[0]

    
def test_hpo_defaults(test_dir, data_frame):
    """

    """
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 10

    # generate some random data
    df = data_frame(feature_col="string_feature",
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    # add categorical feature
    df['categorical_feature'] = ['foo' if r > .5 else 'bar' for r in np.random.rand(n_samples)]

    # add numerical feature
    df['numeric_feature'] = np.random.rand(n_samples)

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(test_dir, "tmp", "real_data_experiment_text_hpo")

    imputer = SimpleImputer(
        input_columns=['string_feature', 'categorical_feature', 'numeric_feature'],
        output_column='label',
        output_path=output_path
    )

    imputer.fit_hpo(df_train, num_evals=2)

    assert imputer.hpo.results.precision_weighted.max() > .7

def test_hpo_many_columns(test_dir, data_frame):
    """

    """
    label_col = "label"

    n_samples = 300
    num_labels = 3
    ncols = 10
    seq_len = 4

    # generate some random data
    df = data_frame(feature_col="string_feature",
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    for col in range(ncols):
        df['string_featur_' + str(col)] = df['string_feature']

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(test_dir, "tmp", "real_data_experiment_text_hpo")

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if not col in ['label']],
        output_column='label',
        output_path=output_path
    )

    imputer.fit_hpo(df_train, num_evals=2)

    assert imputer.hpo.results.precision_weighted.max() > .8


def test_imputer_categorical_heuristic(data_frame):
    """
    Tests the heuristic used for checking whether a column is categorical
    :param data_frame:
    """

    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 20

    # generate some random data
    df = data_frame(feature_col=feature_col,
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    assert SimpleImputer._is_categorical(df[feature_col]) == False
    assert SimpleImputer._is_categorical(df[label_col]) == True


def test_imputer_complete():
    """
    Tests the heuristic used for checking whether a column is categorical
    :param data_frame:
    """

    feature_col = "string_feature"
    label_col = "label"
    feature_col_numeric = "numeric_feature"
    label_col_numeric = "numeric_label"

    num_samples = 1000
    num_labels = 3
    seq_len = 20

    missing_ratio = .1

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

    # delete some entries
    for col in df.columns:
        missing = np.random.random(len(df)) < missing_ratio
        df[col].iloc[missing] = np.nan

    feature_col_missing = df[feature_col].isnull()
    label_col_missing = df[label_col].isnull()

    df = SimpleImputer.complete(data_frame=df)

    assert all(df[feature_col].isnull() == feature_col_missing)
    assert df[label_col].isnull().sum() < label_col_missing.sum()
    assert df[feature_col_numeric].isnull().sum() == 0
    assert df[label_col_numeric].isnull().sum() == 0
