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
from datawig.utils import (logger, rand_string, random_split, generate_df_numeric,
                           generate_df_string)
from datawig import column_encoders
from .conftest import synthetic_label_shift_simple

warnings.filterwarnings("ignore")

logger.setLevel("INFO")


def test_simple_imputer_no_string_column_name():
    with pytest.raises(ValueError):
        SimpleImputer([0], '1')
    with pytest.raises(ValueError):
        SimpleImputer(['0'], 1)


def test_simple_imputer_label_shift(test_dir):
    """
    Test capabilities for detecting and correcting label shift
    """

    tr = synthetic_label_shift_simple(N=1000, label_proportions=[.2, .8], error_proba=.05, covariates=['foo', 'bar'])
    val = synthetic_label_shift_simple(N=500, label_proportions=[.9, .1], error_proba=.05, covariates=['foo', 'bar'])

    # randomly make covariate uninformative
    rand_idxs = np.random.choice(range(val.shape[0]), size=int(val.shape[0]/3), replace=False)
    val.loc[rand_idxs, 'covariate'] = 'foo bar'

    tr, te = random_split(tr, [.8, .2])

    # train domain classifier
    imputer = SimpleImputer(
        input_columns=['covariate'],
        output_column='label',
        output_path=os.path.join(test_dir, "tmp", "label_weighting_experiments"))

    # Fit an imputer model on the train data (coo_imputed_proba, coo_imputed)
    imputer.fit(tr, te, num_epochs=15, learning_rate=3e-4, weight_decay=0)
    pred = imputer.predict(val)

    # compute estimate of ratio of marginals and add corresponding label to the training data
    weights = imputer.check_for_label_shift(val)

    # retrain classifier with balancing
    imputer_balanced = SimpleImputer(
        input_columns=['covariate'],
        output_column='label',
        output_path=os.path.join(test_dir, "tmp", "label_weighting_experiments"))

    # Fit an imputer model on the train data (coo_imputed_proba, coo_imputed)
    imputer_balanced.fit(tr, te, num_epochs=15, learning_rate=3e-4, weight_decay=0, class_weights=weights)

    pred_balanced = imputer_balanced.predict(val)

    acc_balanced = (pred_balanced.label == pred_balanced['label_imputed']).mean()
    acc_classic = (pred.label == pred['label_imputed']).mean()

    # check that weighted performance is better
    assert acc_balanced > acc_classic


def test_label_shift_weight_computation():
    """
    Tests that label shift detection can determine the label marginals of validation data.
    """

    train_proportion = [.7, .3]
    target_proportion = [.3, .7]

    data = synthetic_label_shift_simple(N=2000, label_proportions=train_proportion,
                                        error_proba=.1, covariates=['foo', 'bar'])

    # original train test splits
    tr, te = random_split(data, [.5, .5])

    # train domain classifier
    imputer = SimpleImputer(
        input_columns=['covariate'],
        output_column='label',
        output_path='/tmp/imputer_model')

    # Fit an imputer model on the train data (coo_imputed_proba, coo_imputed)
    imputer.fit(tr, te, num_epochs=15, learning_rate=3e-4, weight_decay=0)

    target_data = synthetic_label_shift_simple(1000, target_proportion,
                                               error_proba=.1, covariates=['foo', 'bar'])

    weights = imputer.check_for_label_shift(target_data)

    # compare the product of weights and training marginals
    # (i.e. estimated target marginals) with the true target marginals.
    for x in list(zip(list(weights.values()), train_proportion, target_proportion)):
        assert x[0]*x[1] - x[2] < .1



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
        train_df=df_train,
        num_epochs=10
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

    retrained_simple_imputer = deserialized.fit(df_train, df_train, num_epochs=10)

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
        num_epochs=10
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
        train_df=df_train,
        num_epochs=10
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

    hps = dict()
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

    imputer_numeric.fit_hpo(df_train, hps=hps)
    results = imputer_numeric.hpo.results

    assert results[results['mse'] == min(results['mse'])]['mse'].iloc[0] < 1.5


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

    hps = dict()
    hps[feature_col] = {}
    hps[feature_col]['type'] = ['string']
    hps[feature_col]['tokens'] = [['words'], ['chars']]

    hps['global'] = {}
    hps['global']['final_fc_hidden_units'] = [[]]
    hps['global']['learning_rate'] = [1e-3]
    hps['global']['weight_decay'] = [0]
    hps['global']['num_epochs'] = [30]

    imputer_string.fit_hpo(df_train, hps=hps, num_epochs=10, num_evals=3)

    assert max(imputer_string.hpo.results['f1_micro']) > 0.7


def test_hpo_all_input_types(test_dir, data_frame):
    """

    Using sklearn advantages: parallelism, distributions of parameters, multiple cross-validation

    """
    label_col = "label"

    n_samples = 500
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

    hps['string_feature'] = {}
    hps['string_feature']['max_tokens'] = [2 ** 15]
    hps['string_feature']['tokens'] = [['words', 'chars']]
    hps['string_feature']['ngram_range'] = {}
    hps['string_feature']['ngram_range']['words'] = [(1, 5)]
    hps['string_feature']['ngram_range']['chars'] = [(2, 4), (1, 3)]

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
                    num_evals=3,
                    hpo_run_name='test1_',
                    num_epochs=10)

    imputer.fit_hpo(df_train,
                    hps=hps,
                    user_defined_scores=uds,
                    num_evals=3,
                    hpo_run_name='test2_',
                    max_running_hours=1/3600,
                    num_epochs=10)

    results = imputer.hpo.results

    assert results[results['global:num_epochs'] == 50]['f1_micro'].iloc[0] > \
           results[results['global:num_epochs'] == 5]['f1_micro'].iloc[0]

    
def test_hpo_defaults(test_dir, data_frame):
    label_col = "label"

    n_samples = 500
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

    imputer.fit_hpo(df_train, num_evals=10, num_epochs=5)

    assert imputer.hpo.results.precision_weighted.max() > .9


def test_hpo_num_evals_empty_hps(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    # generate some random data
    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if col != label_col],
        output_column=label_col,
        output_path=test_dir
    )

    num_evals = 2
    imputer.fit_hpo(df, num_evals=num_evals, num_epochs=10)

    assert imputer.hpo.results.shape[0] == 2


def test_hpo_num_evals_given_hps(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    # generate some random data
    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    # assert that num_evals is an upper bound on the number of hpo runs
    for num_evals in range(1, 3):
        imputer = SimpleImputer(
            input_columns=[col for col in df.columns if col != label_col],
            output_column=label_col,
            output_path=test_dir
        )

        imputer.fit_hpo(df, num_evals=num_evals, num_epochs=5)

        assert imputer.hpo.results.shape[0] == num_evals


def test_hpo_many_columns(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    n_samples = 300
    num_labels = 3
    ncols = 10
    seq_len = 4

    # generate some random data
    df = data_frame(feature_col=feature_col,
                    label_col=label_col,
                    num_labels=num_labels,
                    num_words=seq_len,
                    n_samples=n_samples)

    for col in range(ncols):
        df[feature_col + '_' + str(col)] = df[feature_col]

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if not col in ['label']],
        output_column=label_col,
        output_path=test_dir
    )

    imputer.fit_hpo(df, num_evals=2, num_epochs=10)

    assert imputer.hpo.results.precision_weighted.max() > .75


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


def test_default_no_explainable_simple_imputer():
    imputer = SimpleImputer(
        ['features'],
        'label'
    )
    assert not imputer.is_explainable


def test_explainable_simple_imputer_unfitted():
    label_col = 'label'

    imputer = SimpleImputer(
        ['features'],
        label_col,
        is_explainable=True
    )

    assert imputer.is_explainable

    try:
        imputer.explain('some class')
        raise pytest.fail('imputer.explain should fail with an appropriate error message')
    except ValueError as exception:
        assert exception.args[0] == 'Need to call .fit() before'

    instance = pd.Series({'features': 'some feature text'})
    try:
        imputer.explain_instance(instance)
        raise pytest.fail('imputer.explain_instance should fail with an appropriate error message')
    except ValueError as exception:
        assert exception.args[0] == 'Need to call .fit() before'


def test_explainable_simple_imputer(test_dir, data_frame):
    label_col = 'label'
    df = data_frame(n_samples=100, label_col=label_col)

    output_path = os.path.join(test_dir, "tmp")
    imputer = SimpleImputer(
        ['features'],
        label_col,
        output_path=output_path,
        is_explainable=True
    ).fit(df)

    assert imputer.is_explainable

    assert isinstance(imputer.imputer.data_encoders[0], column_encoders.TfIdfEncoder)

    # explain should not raise an exception
    _ = imputer.explain(df[label_col].unique()[0])

    # explain_instance should not raise an exception
    instance = pd.Series({'features': 'some feature text'})
    _ = imputer.explain_instance(instance)

    assert True


def test_hpo_runs(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if col != label_col],
        output_column=label_col,
        output_path=test_dir
    )

    hps = dict()
    max_tokens = [1024, 2048]
    hps[feature_col] = {'max_tokens': max_tokens}
    hps['global'] = {}
    hps['global']['concat_columns'] = [False]
    hps['global']['num_epochs'] = [10]
    hps['global']['num_epochs'] = [10]
    hps['global']['num_epochs'] = [10]

    imputer.fit_hpo(df, hps=hps, num_hash_bucket_candidates=[2**15], tokens_candidates=['words'])

    # only search over specified parameter ranges
    assert set(imputer.hpo.results[feature_col+':'+'max_tokens'].unique().tolist()) == set(max_tokens)
    assert imputer.hpo.results.shape[0] == 2


def test_hpo_single_column_encoder_parameter(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if col != label_col],
        output_column=label_col,
        output_path=test_dir,
        is_explainable=True
    )

    hps = dict()
    hps[feature_col] = {'max_tokens': [1024]}
    hps['global'] = {}
    hps['global']['num_epochs'] = [10]

    imputer.fit_hpo(df, hps=hps)

    assert imputer.hpo.results.shape[0] == 2
    assert imputer.imputer.data_encoders[0].vectorizer.max_features == 1024


def test_hpo_multiple_columns_only_one_used(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)
    df.loc[:, feature_col+'_2'] = df.loc[:, feature_col]

    imputer = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=test_dir,
        is_explainable=True
    )

    hps = dict()
    hps['global'] = {}
    hps['global']['num_epochs'] = [10]
    hps[feature_col] = {'max_tokens': [1024]}
    hps[feature_col]['tokens'] = [['chars']]

    imputer.fit_hpo(df, hps=hps)

    assert imputer.hpo.results.shape[0] == 1
    assert imputer.imputer.data_encoders[0].vectorizer.max_features == 1024


def test_hpo_mixed_hps_and_kwargs(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=test_dir
    )

    hps = {feature_col: {'max_tokens': [1024]}}

    imputer.fit_hpo(df, hps=hps, learning_rate_candidates=[0.1])

    assert imputer.hpo.results['global:learning_rate'].values[0] == 0.1


def test_hpo_mixed_hps_and_kwargs_precedence(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=test_dir
    )

    hps = {feature_col: {'max_tokens': [1024]}, 'global': {'learning_rate': [0.11]}}

    imputer.fit_hpo(df, hps=hps, learning_rate_candidates=[0.1])

    # give parameters in `hps` precedence over fit_hpo() kwargs
    assert imputer.hpo.results['global:learning_rate'].values[0] == 0.11


def test_hpo_similar_input_col_mixed_types(test_dir, data_frame):
    feature_col, label_col = "feature", "label"
    numeric_col = "numeric_feature"
    categorical_col = "categorical_col"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    df.loc[:, numeric_col] = np.random.randn(df.shape[0])
    df.loc[:, categorical_col] = np.random.randint(df.shape[0])

    imputer = SimpleImputer(
        input_columns=[feature_col, numeric_col, categorical_col],
        output_column=label_col,
        output_path=test_dir
    )

    imputer.fit_hpo(df, num_epochs=10)


def test_hpo_kwargs_only_support(test_dir, data_frame):
    feature_col, label_col = "feature", "label"
    numeric_col = "numeric_feature"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    df.loc[:, numeric_col] = np.random.randn(df.shape[0])

    imputer = SimpleImputer(
        input_columns=[feature_col, numeric_col],
        output_column=label_col,
        output_path=test_dir
    )

    imputer.fit_hpo(
        df,
        num_epochs=1,
        patience=1,
        weight_decay=[0.001],
        batch_size=320,
        num_hash_bucket_candidates=[3],
        tokens_candidates=['words'],
        numeric_latent_dim_candidates=[1],
        numeric_hidden_layers_candidates=[1],
        final_fc_hidden_units=[[1]],
        learning_rate_candidates=[0.1],
        normalize_numeric=False
    )

    def assert_val(col, value):
        assert imputer.hpo.results[col].values[0] == value

    assert_val('global:num_epochs', 1)
    assert_val('global:patience', 1)
    assert_val('global:weight_decay', 0.001)

    assert_val('global:batch_size', 320)
    assert_val(feature_col + ':max_tokens', 3)
    assert_val(feature_col + ':tokens', ['words'])

    assert_val(numeric_col + ':numeric_latent_dim', 1)
    assert_val(numeric_col + ':numeric_hidden_layers', 1)

    assert_val('global:final_fc_hidden_units', [1])
    assert_val('global:learning_rate', 0.1)


def test_hpo_numeric_best_pick(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    df.loc[:, label_col] = np.random.randn(df.shape[0])

    imputer = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=test_dir,
        is_explainable=True
    )

    hps = {feature_col: {'max_tokens': [1, 2, 3]}}
    hps[feature_col]['tokens'] = [['chars']]

    imputer.fit_hpo(df, hps=hps)

    results = imputer.hpo.results

    max_tokens_of_encoder = imputer.imputer.data_encoders[0].vectorizer.max_features

    # model with minimal MSE
    best_hpo_run = imputer.hpo.results['mse'].astype('float').idxmin()
    loaded_hpo_run = results.loc[results[feature_col+':max_tokens'] == max_tokens_of_encoder].index[0]

    assert best_hpo_run == loaded_hpo_run


def test_fit_resumes(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=test_dir
    )

    assert imputer.imputer is None

    imputer.fit(df)
    first_fit_imputer = imputer.imputer

    imputer.fit(df)
    second_fit_imputer = imputer.imputer

    assert first_fit_imputer == second_fit_imputer


def test_hpo_explainable(test_dir, data_frame):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    for explainable, vectorizer in [(False, HashingVectorizer), (True, TfidfVectorizer)]:
        imputer = SimpleImputer(
            input_columns=[feature_col],
            output_column=label_col,
            output_path=test_dir,
            is_explainable=explainable
        ).fit_hpo(df, num_epochs=3)
        assert isinstance(imputer.imputer.data_encoders[0].vectorizer, vectorizer)
