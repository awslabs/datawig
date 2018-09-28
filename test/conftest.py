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

import os
import random

import shutil

import mxnet as mx
import numpy as np
import pandas as pd
import pytest

from datawig.utils import rand_string


@pytest.fixture(scope='function')
def reset_random_seed():
    # Setting seed for PRNG(s) for every test in the file
    random.seed(0)
    np.random.seed(0)
    mx.random.seed(0)

    yield

@pytest.fixture
def test_dir():
    file_dir = os.path.dirname(os.path.realpath(__file__))
    joined = os.path.join(file_dir, "resources")
    yield joined
    shutil.rmtree(joined)

@pytest.fixture
def data_frame():

    def _inner_impl(
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

        def _sentence_with_label(labels=labels, words=words):
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

        sentences, labels = zip(*[_sentence_with_label(labels, words) for _ in range(n_samples)])
        df = pd.DataFrame({feature_col: sentences, label_col: labels})

        return df

    return _inner_impl
