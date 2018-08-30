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
import random
import warnings
import shutil
import numpy as np
import pandas as pd
import mxnet as mx
from sklearn.metrics import f1_score, mean_squared_error

dir_path = os.path.dirname(os.path.realpath(__file__))

from datawig.utils import rand_string, random_split, logger
from datawig.column_encoders import BowEncoder
from datawig.mxnet_input_symbols import BowFeaturizer
from datawig.simple_imputer import SimpleImputer

from test_imputer import generate_string_data_frame, create_test_image

warnings.filterwarnings("ignore")

logger.setLevel("INFO")

mx.random.seed(1)
np.random.seed(42)
random.seed(1)


def test_simple_imputer_real_data_default_args():
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
    random_data = generate_string_data_frame(feature_col=feature_col,
                                             label_col=label_col,
                                             vocab_size=vocab_size,
                                             num_labels=num_labels,
                                             num_words=seq_len,
                                             n_samples=n_samples)

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment_simple")

    df_train_cols_before = df_train.columns.tolist()

    input_columns = [feature_col]

    imputer = SimpleImputer(
        input_columns=input_columns,
        output_column=label_col,
        output_path=output_path
    ).fit(
        train_df=df_train
    )

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

    df_test_imputed = imputer.predict(df_no_label_column)

    assert all(
        [after == before for after, before in zip(df_no_label_column.columns, df_test_cols_before)])

    imputed_columns = df_test_cols_before + [label_col + "_imputed", label_col + "_imputed_proba"]

    assert all([after == before for after, before in zip(df_test_imputed, imputed_columns)])

    f1 = f1_score(true_labels, df_test_imputed[label_col + '_imputed'], average="weighted")

    assert f1 > .9

    new_path = imputer.output_path + "-" + rand_string()

    if os.path.isdir(new_path): shutil.rmtree(new_path)

    os.rename(imputer.output_path, new_path)

    deserialized = SimpleImputer.load(new_path)
    df_test = deserialized.predict(df_test, imputation_suffix="_deserialized_imputed")
    f1 = f1_score(df_test[label_col], df_test[label_col + '_deserialized_imputed'],
                  average="weighted")

    assert f1 > .9

    retrained_simple_imputer = deserialized.fit(df_train, df_train)

    df_train_imputed = retrained_simple_imputer.predict(df_train.copy())
    f1 = f1_score(df_train[label_col], df_train_imputed[label_col + '_imputed'], average="weighted")

    assert f1 > .9

    metrics = retrained_simple_imputer.load_metrics()

    assert f1 == metrics['weighted_f1']

    shutil.rmtree(output_path)


def test_numeric_or_text_imputer():
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
    random_data = generate_string_data_frame(feature_col=feature_col,
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
    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment_numeric")

    imputer_numeric_linear = SimpleImputer(
        input_columns=['x', feature_col],
        output_column="*2",
        output_path=output_path
    ).fit(
        train_df=df_train,
        learning_rate=1e-3,
    )

    imputer_numeric_linear.predict(df_test)

    assert mean_squared_error(df_test['*2'], df_test['*2_imputed']) < 1.0

    imputer_numeric = SimpleImputer(
        input_columns=['x', feature_col],
        output_column="**2",
        output_path=output_path
    ).fit(
        train_df=df_train,
        learning_rate=1e-3
    )

    imputer_numeric.predict(df_test)

    assert mean_squared_error(df_test['**2'], df_test['**2_imputed']) < 1.0

    imputer_string = SimpleImputer(
        input_columns=[feature_col, 'x'],
        output_column=label_col,
        output_path=output_path
    ).fit(
        train_df=df_train
    )

    imputer_string.predict(df_test)

    assert f1_score(df_test[label_col], df_test[label_col + '_imputed'], average="weighted") > .7

    shutil.rmtree(output_path)


def test_imputer_hpo_numeric():
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
    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment_numeric_hpo")

    imputer_numeric = SimpleImputer(
        input_columns=['x'],
        output_column="**2",
        output_path=output_path
    ).fit_hpo(
        train_df=df_train,
        learning_rate=1e-3,
        num_epochs=100,
        patience=10,
        num_hash_bucket_candidates=[2 ** 10],
        tokens_candidates=['words'],
        latent_dim_candidates=[10, 50, 100],
        hidden_layers_candidates=[1, 2]
    )

    imputer_numeric.predict(df_test)

    assert mean_squared_error(df_test['**2'], df_test['**2_imputed']) < 1.0

    shutil.rmtree(output_path)


def test_imputer_hpo_text():
    """

    Tests SimpleImputer HPO with text data and categorical imputations

    """
    feature_col = "string_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 3
    seq_len = 20

    # generate some random data
    df = generate_string_data_frame(feature_col=feature_col,
                                    label_col=label_col,
                                    num_labels=num_labels,
                                    num_words=seq_len,
                                    n_samples=n_samples)

    df_train, df_test = random_split(df, [.8, .2])
    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment_text_hpo")

    imputer_string = SimpleImputer(
        input_columns=[feature_col],
        output_column=label_col,
        output_path=output_path
    ).fit_hpo(
        train_df=df_train,
        num_epochs=100,
        patience=3,
        num_hash_bucket_candidates=[2 ** 10, 2 ** 15],
        tokens_candidates=['words'],
        latent_dim_candidates=[10],
        hpo_max_train_samples=1000
    )

    imputer_string.predict(df_test)

    assert f1_score(df_test[label_col], df_test[label_col + '_imputed'], average="weighted") > .7

    shutil.rmtree(output_path)


def test_imputer_image_hpo():
    """

    Tests SimpleImputer HPO with image data imputing a text column

    """

    img_path = os.path.join(dir_path, "resources", "test_images")
    os.makedirs(img_path, exist_ok=True)

    colors = ['red', 'green', 'blue']

    for color in colors:
        create_test_image(os.path.join(img_path, color + ".png"), color)

    n_samples = 32
    color_labels = [random.choice(colors) for _ in range(n_samples)]

    df = pd.DataFrame({"image_files": color_labels,
                       "label": color_labels})

    df['image_files'] = img_path + "/" + df['image_files'] + ".png"

    output_path = os.path.join(dir_path, "resources", "tmp", "experiment_images_hpo")

    imputer_string = SimpleImputer(
        input_columns=['image_files'],
        output_column="label",
        output_path=output_path
    ).fit_hpo(
        train_df=df,
        learning_rate=1e-3,
        num_epochs=10,
        patience=10,
        test_split=.3,
        weight_decay=.0,
        num_hash_bucket_candidates=[2 ** 10],
        tokens_candidates=['words'],
        latent_dim_candidates=[10, 100],
        hpo_max_train_samples=1000
    )
