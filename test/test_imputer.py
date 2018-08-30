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

DataWig imputer tests

"""

import os
import shutil
import random
import pytest
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import mxnet as mx

from io import BytesIO
from PIL import Image

dir_path = os.path.dirname(os.path.realpath(__file__))

mx.random.seed(1)
random.seed(1)
np.random.seed(42)

from datawig.column_encoders import SequentialEncoder, CategoricalEncoder, BowEncoder, \
    NumericalEncoder, ImageEncoder
from datawig.mxnet_input_symbols import LSTMFeaturizer, EmbeddingFeaturizer, BowFeaturizer, \
    NumericalFeaturizer, ImageFeaturizer
from datawig.imputer import Imputer
from datawig.utils import random_split, rand_string


def create_test_image(filename, color='red'):
    if color == 'red':
        color_vector = (155, 0, 0)
    elif color == 'green':
        color_vector = (0, 155, 0)
    elif color == 'blue':
        color_vector = (0, 0, 155)
    file = BytesIO()
    image = Image.new('RGBA', size=(50, 50), color=color_vector)
    image.save(filename, 'png')
    file.name = filename + '.png'
    file.seek(0)
    return file

def generate_string_data_frame(
        feature_col='features',
        label_col='labels',
        n_samples=500,
        word_length=5,
        num_words=100,
        vocab_size=100,
        num_labels=10):
    """
    Generates text features and categorical labels.
    :param feature_col: name of feature column.
    :param label_col: name of label column.
    :param n_samples: how many rows to generate.
    :return: pd.DataFrame with columns = [feature_col, label_col]
    """

    vocab = [rand_string(word_length) for i in range(vocab_size)]
    labels = vocab[:num_labels]
    words = vocab[num_labels:]

    def sentence_with_label(labels=labels, words=words):
        """
        Generates a random token sequence containing a random label

        :param labels: label set
        :param words: vocabulary of tokens
        :return: blank separated token sequence and label

        """
        label = random.choice(labels)
        tokens = [random.choice(words) for _ in range(num_words)] + [label]
        sentence = " ".join(np.random.permutation(tokens))
        return sentence, label

    sentences, labels = zip(*[sentence_with_label(labels, words) for _ in range(n_samples)])

    df = pd.DataFrame({feature_col: sentences, label_col: labels})

    return df


def test_drop_missing():
    """
    Tests some private functions of the Imputer class
    """
    df_train = pd.DataFrame(
        {'label': [1, None, np.nan, 2] * 4, 'data': ['bla', 'drop', 'drop', 'fasl'] * 4})
    df_test = df_train.copy()

    max_tokens = int(2 ** 15)

    batch_size = 16

    data_encoder_cols = [BowEncoder('data', max_tokens=max_tokens)]
    label_encoder_cols = [CategoricalEncoder('label', max_tokens=1)]
    data_cols = [BowFeaturizer('data', vocab_size=max_tokens)]

    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment")

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df_train,
        test_df=df_test,
        batch_size=batch_size
    )

    df_dropped = imputer._Imputer__drop_missing_labels(df_train, how='any')

    df_dropped_true = pd.DataFrame({'data': {3: 'fasl', 7: 'fasl', 11: 'fasl', 15: 'fasl'},
                                    'label': {3: 2.0, 7: 2.0, 11: 2.0, 15: 2.0}})

    assert df_dropped[['data', 'label']].equals(df_dropped_true[['data', 'label']])


def test_imputer_init():
    with pytest.raises(ValueError) as e:
        imputer = Imputer(data_featurizers='item_name', label_encoders=['brand'], data_encoders='')

    with pytest.raises(ValueError) as e:
        imputer = Imputer(data_featurizers=[BowFeaturizer('item_name')],
                          label_encoders="brand",
                          data_encoders='')

    with pytest.raises(ValueError) as e:
        imputer = Imputer(data_featurizers=[BowFeaturizer('item_name')],
                          label_encoders=[CategoricalEncoder("brand")],
                          data_encoders='')

    with pytest.raises(ValueError) as e:
        imputer = Imputer(data_featurizers=[BowFeaturizer('item_name')],
                          label_encoders=[CategoricalEncoder("brand")],
                          data_encoders=[BowEncoder('not_in_featurizers')])

    with pytest.raises(ValueError) as e:
        imputer = Imputer(data_featurizers=[BowFeaturizer('item_name')],
                          label_encoders=[CategoricalEncoder("brand")],
                          data_encoders=[BowEncoder('brand')])

    label_encoders = [CategoricalEncoder('brand', max_tokens=10)]
    data_featurizers = [LSTMFeaturizer('item_name'), EmbeddingFeaturizer('manufacturer')]

    data_encoders = [
        SequentialEncoder(
            'item_name'
        ),
        CategoricalEncoder(
            'manufacturer'
        )
    ]

    imputer = Imputer(
        data_featurizers=data_featurizers,
        label_encoders=label_encoders,
        data_encoders=data_encoders
    )

    assert imputer.output_path == "brand"
    assert imputer.module_path == 'brand/model'
    assert imputer.metrics_path == 'brand/fit-test-metrics.json'

    assert imputer.output_path == "brand"
    assert imputer.module_path == 'brand/model'
    assert imputer.metrics_path == 'brand/fit-test-metrics.json'

    imputer = Imputer(
        data_featurizers=data_featurizers,
        label_encoders=[CategoricalEncoder('B Rand', max_tokens=10)],
        data_encoders=data_encoders
    )
    assert imputer.output_path == "b_rand"

    shutil.rmtree("b_rand")

def test_imputer_duplicate_encoder_output_columns():
    """
    Tests Imputer with sequential, bag-of-words and categorical variables as inputs
    this could be run as part of integration test suite.
    """

    feature_col = "string_feature"
    categorical_col = "categorical_feature"
    label_col = "label"

    n_samples = 1000
    num_labels = 10
    seq_len = 100
    vocab_size = int(2 ** 10)

    latent_dim = 30
    embed_dim = 30

    # generate some random data
    random_data = generate_string_data_frame(feature_col=feature_col,
                                             label_col=label_col,
                                             vocab_size=vocab_size,
                                             num_labels=num_labels,
                                             num_words=seq_len,
                                             n_samples=n_samples)

    # we use a the label prefixes as a dummy categorical input variable
    random_data[categorical_col] = random_data[label_col].apply(lambda x: x[:2])

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    data_encoder_cols = [
        BowEncoder(feature_col, feature_col, max_tokens=vocab_size),
        SequentialEncoder(feature_col, feature_col, max_tokens=vocab_size, seq_len=seq_len),
        CategoricalEncoder(categorical_col, max_tokens=num_labels)
    ]
    label_encoder_cols = [CategoricalEncoder(label_col, max_tokens=num_labels)]

    data_cols = [
        BowFeaturizer(
            feature_col,
            vocab_size=vocab_size),
        LSTMFeaturizer(
            field_name=feature_col,
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_hidden=30,
            embed_dim=embed_dim,
            num_layers=2,
            vocab_size=num_labels),
        EmbeddingFeaturizer(
            field_name=categorical_col,
            embed_dim=embed_dim,
            vocab_size=num_labels)
    ]

    output_path = os.path.join(dir_path, "resources", "tmp",
                               "imputer_experiment_synthetic_data")

    num_epochs = 20
    batch_size = 16
    learning_rate = 1e-3

    with pytest.raises(ValueError) as e:
        imputer = Imputer(
            data_featurizers=data_cols,
            label_encoders=label_encoder_cols,
            data_encoders=data_encoder_cols,
            output_path=output_path
        ).fit(
            train_df=df_train,
            test_df=df_val,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
        shutil.rmtree(output_path)


def test_imputer_real_data_all_featurizers():
    """
    Tests Imputer with sequential, bag-of-words and categorical variables as inputs
    this could be run as part of integration test suite.
    """

    feature_col = "string_feature"
    categorical_col = "categorical_feature"
    label_col = "label"

    n_samples = 5000
    num_labels = 3
    seq_len = 20
    vocab_size = int(2 ** 10)

    latent_dim = 30
    embed_dim = 30

    # generate some random data
    random_data = generate_string_data_frame(feature_col=feature_col,
                                             label_col=label_col,
                                             vocab_size=vocab_size,
                                             num_labels=num_labels,
                                             num_words=seq_len,
                                             n_samples=n_samples)

    # we use a the label prefixes as a dummy categorical input variable
    random_data[categorical_col] = random_data[label_col].apply(lambda x: x[:2])

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    data_encoder_cols = [
        BowEncoder(feature_col, feature_col + "_bow", max_tokens=vocab_size),
        SequentialEncoder(feature_col, feature_col + "_lstm", max_tokens=vocab_size, seq_len=seq_len),
        CategoricalEncoder(categorical_col, max_tokens=num_labels)
    ]
    label_encoder_cols = [CategoricalEncoder(label_col, max_tokens=num_labels)]

    data_cols = [
        BowFeaturizer(
            feature_col + "_bow",
            vocab_size=vocab_size),
        LSTMFeaturizer(
            field_name=feature_col + "_lstm",
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_hidden=30,
            embed_dim=embed_dim,
            num_layers=2,
            vocab_size=num_labels),
        EmbeddingFeaturizer(
            field_name=categorical_col,
            embed_dim=embed_dim,
            vocab_size=num_labels)
    ]

    output_path = os.path.join(dir_path, "resources", "tmp", "imputer_experiment_synthetic_data")

    num_epochs = 10
    batch_size = 32
    learning_rate = 1e-2

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df_train,
        test_df=df_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    len_df_before_predict = len(df_test)
    pred = imputer.transform(df_test)

    assert len(pred[label_col]) == len_df_before_predict

    assert sum(df_test[label_col].values == pred[label_col]) == len(df_test)

    _ = imputer.predict_proba_top_k(df_test, top_k=2)

    _, metrics = imputer.transform_and_compute_metrics(df_test)

    assert metrics[label_col]['avg_f1'] > 0.9

    deserialized = Imputer.load(imputer.output_path)

    _, metrics_deserialized = deserialized.transform_and_compute_metrics(df_test)

    assert metrics_deserialized[label_col]['avg_f1'] > 0.9

    # training on a small data set to get a imputer with low precision
    not_so_precise_imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df_train[:50],
        test_df=df_test,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    df_test = df_test.reset_index()
    predictions_df = not_so_precise_imputer.predict(df_test, precision_threshold=.9,
                                                    imputation_suffix="_imputed")

    assert predictions_df.columns.contains(label_col + "_imputed")
    assert predictions_df.columns.contains(label_col + "_imputed_proba")
    assert predictions_df.loc[0, label_col + '_imputed'] == df_test.loc[0, label_col]
    assert np.isnan(predictions_df.loc[0, label_col + '_imputed_proba']) == False
    assert len(predictions_df.dropna(subset=[label_col + "_imputed_proba"])) < n_samples

    shutil.rmtree(output_path)

def test_imputer_without_train_df():
    """
    Test asserting that imputer.fit fails without training data or training data in wrong format
    """
    df_train = ['ffffffooooo']

    data_encoder_cols = [
        BowEncoder('item_name')
    ]
    label_encoder_cols = [CategoricalEncoder('brand')]

    data_cols = [
        BowFeaturizer('item_name')
    ]

    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment")

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path,
    )

    with pytest.raises(ValueError, message="Need a non-empty DataFrame for fitting Imputer model"):
        imputer.fit(
            train_df=df_train
        )

    with pytest.raises(ValueError, message="Need a non-empty DataFrame for fitting Imputer model"):
        imputer.fit(
            train_df=None
        )


def test_imputer_without_test_set_random_split():
    """
    Test asserting that the random split is working internally
    by calling imputer.fit only with a training set.
    """

    feature_col = "string_feature"
    label_col = "label"

    n_samples = 5000
    num_labels = 3
    seq_len = 20
    vocab_size = int(2 ** 10)

    # generate some random data
    df_train = generate_string_data_frame(feature_col=feature_col,
                                             label_col=label_col,
                                             vocab_size=vocab_size,
                                             num_labels=num_labels,
                                             num_words=seq_len,
                                             n_samples=n_samples)


    num_epochs = 1
    batch_size = 64
    learning_rate = 1e-3

    data_encoder_cols = [
        BowEncoder(feature_col, max_tokens=vocab_size)
    ]
    label_encoder_cols = [CategoricalEncoder(label_col, max_tokens=num_labels)]

    data_cols = [
        BowFeaturizer(feature_col, vocab_size=vocab_size)
    ]

    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment")

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    )

    try:
        imputer.fit(
            train_df=df_train,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            batch_size=batch_size
        )
    except TypeError:
        pytest.fail("Didn't expect a TypeError exception with missing test data")

    shutil.rmtree(output_path)

def test_imputer_load_read_exec_only_dir(tmpdir):
    import stat

    # on shared build-fleet tests fail with converting tmpdir to string
    tmpdir = str(tmpdir)
    feature = 'feature'
    label = 'label'

    df = generate_string_data_frame(feature, label, n_samples=100)
    # fit and output model + metrics to tmpdir

    imputer = Imputer(
        data_featurizers=[BowFeaturizer(feature)],
        label_encoders=[CategoricalEncoder(label)],
        data_encoders=[BowEncoder(feature)],
        output_path=tmpdir
    ).fit(train_df=df, num_epochs=1)

    # make tmpdir read/exec-only by owner/group/others
    os.chmod(tmpdir,
             stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH | stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    try:
        Imputer.load(tmpdir)
    except AssertionError as e:
        print(e)
        pytest.fail('Loading imputer from read-only directory should not fail.')

def test_imputer_fit_fail_non_writable_output_dir(tmpdir):
    import stat

    # on shared build-fleet tests fail with converting tmpdir to string
    tmpdir = str(tmpdir)
    feature = 'feature'
    label = 'label'
    df = generate_string_data_frame(feature, label, n_samples=100)
    # fit and output model + metrics to tmpdir
    imputer = Imputer(
        data_featurizers=[BowFeaturizer(feature)],
        label_encoders=[CategoricalEncoder(label)],
        data_encoders=[BowEncoder(feature)],
        output_path=tmpdir
    )

    # make tmpdir read/exec-only by owner/group/others
    os.chmod(tmpdir,
             stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH | stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)

    # fail if imputer.fit does not raise an AssertionError
    with pytest.raises(AssertionError) as e:
        imputer.fit(df, num_epochs=1)


def test_imputer_numeric_data():
    """
    Tests numeric encoder/featurizer only

    """
    # Training data
    N = 1000
    x = np.random.uniform(-np.pi, np.pi, (N,))
    df = pd.DataFrame({
        'x': x,
        'cos': np.cos(x),
        '*2': x * 2,
        '**2': x ** 2})

    df_train, df_test = random_split(df, [.6, .4])
    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment_numeric")

    data_encoder_cols = [NumericalEncoder(['x'])]
    data_cols = [NumericalFeaturizer('x', latent_dim=100)]

    for target in ['*2', '**2', 'cos']:
        label_encoder_cols = [NumericalEncoder([target], normalize=False)]

        imputer = Imputer(
            data_featurizers=data_cols,
            label_encoders=label_encoder_cols,
            data_encoders=data_encoder_cols,
            output_path=output_path
        ).fit(
            train_df=df_train,
            learning_rate=1e-1,
            num_epochs=100,
            patience=5,
            test_split=.3,
            weight_decay=.0,
            batch_size=128
        )

        pred, metrics = imputer.transform_and_compute_metrics(df_test)
        df_test['predictions_' + target] = pred[target].flatten()
        print("Numerical metrics: {}".format(metrics[target]))
        assert metrics[target] < 10

        shutil.rmtree(output_path)


def test_imputer_image_data():

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

    output_path = os.path.join(dir_path, "resources", "tmp", "experiment_images")

    data_encoder_cols = [ImageEncoder(['image_files'])]
    data_cols = [ImageFeaturizer('image_files')]

    label_encoder_cols = [CategoricalEncoder(['label'])]

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df,
        learning_rate=1e-3,
        num_epochs=2,
        patience=5,
        test_split=.1,
        weight_decay=.0001,
        batch_size=16
    )

    shutil.rmtree(output_path)

    # Test with image + numeric inputs
    df['numeric'] = np.random.uniform(-np.pi, np.pi, (n_samples,))

    output_path = os.path.join(dir_path, "resources", "tmp", "experiment_images_with_num")

    data_encoder_cols = [ImageEncoder(['image_files']), NumericalEncoder(['numeric'])]
    data_cols = [ImageFeaturizer('image_files'), NumericalFeaturizer('numeric', latent_dim=100)]
    label_encoder_cols = [CategoricalEncoder(['label'])]

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df,
        learning_rate=1e-3,
        num_epochs=2,
        patience=5,
        test_split=.1,
        weight_decay=.0001,
        batch_size=16
    )
    shutil.rmtree(img_path)
    shutil.rmtree(output_path)


def test_imputer_unrepresentative_test_df():
    """

    Tests whether the imputer runs through in cases when test data set (and hence metrics and precision/recall curves)
    doesn't contain values present in training data

    """
    # generate some random data
    random_data = generate_string_data_frame(n_samples=100)

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    excluded = df_train['labels'].values[0]
    df_test = df_test[df_test['labels'] != excluded]

    data_encoder_cols = [BowEncoder('features')]
    label_encoder_cols = [CategoricalEncoder('labels')]
    data_cols = [BowFeaturizer('features')]

    output_path = os.path.join(dir_path, "resources", "tmp", "real_data_experiment")

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df_train,
        test_df=df_test,
        num_epochs=10)

    only_excluded_df = df_train[df_train['labels'] == excluded]
    imputations = imputer.predict_above_precision(only_excluded_df,
                                                  precision_threshold=.99)['labels']
    assert all([x == () for x in imputations])
    shutil.rmtree(output_path)