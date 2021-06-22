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

DataWig utility functions

"""

import contextlib
import itertools
import logging
import math
import random
import sys
import time
import string
import collections
from typing import Any, List, Tuple, Dict

import numpy as np
import pandas as pd

random.seed(1)
np.random.seed(42)

class ColumnOverwriteException(Exception):
    """Raised when an existing column of a pandas dataframe is about to be overwritten"""
    pass

class TargetSparsityException(Exception):
    """Raised when a target column cannot be used as label for a supervised learning model"""
    pass

def random_split(data_frame: pd.DataFrame,
                 split_ratios: List[float] = None,
                 seed: int = 10) -> List[pd.DataFrame]:
    """

    Shuffles and splits a Data frame into partitions with specified percentages of data

    :param data_frame: a pandas DataFrame
    :param split_ratios: percentages of splits
    :param seed: seed of random number generator
    :return:
    """
    if split_ratios is None:
        split_ratios = [.8, .2]
    sections = np.array([int(r * len(data_frame)) for r in split_ratios]).cumsum()
    return np.split(data_frame.sample(frac=1, random_state=seed), sections)[:len(split_ratios)]


def rand_string(length: int = 16) -> str:
    """
    Utility function for generating a random alphanumeric string of specified length

    :param length: length of the generated string

    :return: random string
    """
    import random, string
    return ''.join([random.choice(string.ascii_letters + string.digits) for _ in range(length)])

def normalize_dataframe(data_frame: pd.DataFrame):
    """

    Convenience function that normalizes strings of a DataFrame by converting to lower-case,
    stripping spaces, keeping only alphanumeric and space.

    :param data_frame:
    :return: normalized data_frame
    """
    return data_frame.apply(
        lambda x: x
            .astype(str)
            .str.lower()
            .str.replace(r'\s+', ' ')
            .str.strip()
            .str.replace('[^a-zA-Z0-9_ \r\n\t\f\v]', '')
    )

def generate_df_string(word_length: int = 5,
                       vocab_size: int = 100,
                       num_labels: int = 2,
                       num_words: int = 5,
                       num_samples: int = 200,
                       label_column_name: str = "labels",
                       data_column_name: str = "sentences") -> pd.DataFrame:
    """
    Generates a dataframe with random strings in one column and random 'labels', which are
     substrings contained in the string column.

     Use this method for testing the imputer on string data

    :param word_length: length of the synthetic words
    :param vocab_size:  size of synthetic vocabulary
    :param num_labels:  number of labels / categories
    :param num_words:   number of words in each sentence
    :param n_samples:   number of samples in the data frame
    :param label_column_name: name of the label column
    :param data_column_name:  name of the data column
    :return:

    """
    vocab = [rand_string(word_length) for _ in range(vocab_size)]
    labels, words = vocab[:num_labels], vocab[num_labels:]

    def sentence_with_label(labels=labels, words=words):
        label = random.choice(labels)
        return " ".join(np.random.permutation([random.choice(words) for _ in range(num_words)] + [label])), label

    sentences, labels = zip(*[sentence_with_label(labels, words) for _ in range(num_samples)])

    return pd.DataFrame({data_column_name: sentences, label_column_name: labels})


def generate_df_numeric(num_samples: int = 100,
                        label_column_name: str = "f(x)",
                        data_column_name: str = "x") -> pd.DataFrame:
    """
    Generates a dataframe with random numbers between -pi and pi in one column and the square of those values in another

    :param num_samples:         number of samples to be generated
    :param label_column_name:   name of label column
    :param data_column_name:    name of data column
    :return:
    """
    numeric_data = np.random.uniform(-np.pi, np.pi, (num_samples,))
    return pd.DataFrame({
        data_column_name: numeric_data,
        label_column_name: numeric_data ** 2 + np.random.normal(0, .01, (num_samples,)),
    })
