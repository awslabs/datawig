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

DataWig calibration tests

"""
import numpy as np

from datawig import calibration
from datawig.column_encoders import (BowEncoder, CategoricalEncoder,
                                     SequentialEncoder)
from datawig.imputer import Imputer
from datawig.mxnet_input_symbols import (BowFeaturizer, EmbeddingFeaturizer,
                                         LSTMFeaturizer)
from datawig.utils import random_split


def generate_synthetic_data(K: int=5, N: int=100, p_correct: float=.8):
    """
    Generate synthetic data of class probabilities

    :param K: number of classes
    :param N: number of samples
    :param p_correct: fraction of correct predictions
    :return: tuple of training data and training labels
    """

    # generate labels
    train_labels = np.array([np.random.randint(5) for n in range(N)])

    # generate features
    train_data = np.empty([N, K])

    for n in range(N):
        # pick correct label with probability p_correct
        if np.random.rand() < p_correct:
            label = train_labels[n]
        else:
            label = np.random.choice([val for val in range(K) if val != train_labels[n]])

        # assign logits from uniform [1,2] for correct and uniform [0,1] for incorrect labels
        for k in range(K):
            if label == k:
                train_data[n, k] = np.random.uniform(1, 2)
            else:
                train_data[n, k] = np.random.uniform(0, 1)

        train_data[n, :] = calibration.softmax(train_data[n, :])

    # assert probabilities sum to 1
    assert np.all((np.sum(train_data, 1) - 1) < 1e-10)

    return train_data, train_labels


def test_calibration_synthetic():
    """
    For simple training data, fit the temperature scaling model and assert that the
    expected calibration error is reduced.
    """
    train_data, train_labels = generate_synthetic_data(p_correct=.7, N=50, K=10)

    temperature = calibration.fit_temperature(train_data, train_labels)

    assert calibration.compute_ece(train_data, train_labels, lbda=temperature) < \
           calibration.compute_ece(train_data, train_labels)


def test_automatic_calibration(data_frame):
    """
    Fit model with all featurisers and assert
    that calibration improves the expected calibration error.
    """

    feature_col = "string_feature"
    categorical_col = "categorical_feature"
    label_col = "label"

    n_samples = 2000
    num_labels = 3
    seq_len = 20
    vocab_size = int(2 ** 10)

    latent_dim = 30
    embed_dim = 30

    # generate some random data
    random_data = data_frame(feature_col=feature_col,
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

    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-2

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols
    ).fit(
        train_df=df_train,
        test_df=df_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    assert imputer.calibration_info['ece_pre'] > imputer.calibration_info['ece_post']
