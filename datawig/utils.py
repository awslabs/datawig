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

import mxnet as mx
import numpy as np
import pandas as pd

mx.random.seed(1)
random.seed(1)
np.random.seed(42)

# set global logger variables
log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
logger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
consoleHandler.setLevel("INFO")
logger.addHandler(consoleHandler)
logger.setLevel("INFO")


def set_stream_log_level(level: str):
    for handler in logger.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(level)


def flatten_dict(d: Dict,
                 parent_key: str ='',
                 sep: str =':') -> Dict:
    """
    Flatten a nested dictionary and create new keys by concatenation

    :param d: input dictionary (nested)
    :param parent_key: Prefix for keys of the flat dictionary
    :param sep: Separator when concatenating dictionary keys from different levels.
    """

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class ColumnOverwriteException(Exception):
    """Raised when an existing column of a pandas dataframe is about to be overwritten"""
    pass


def stringify_list(cols):
    """
    Returns list with elements stringified
    """
    return [str(c) for c in cols]


def merge_dicts(d1: dict, d2: dict):
    """

    Merges two dicts

    :param d1: A dictionary
    :param d2: Another dictionary
    :return: Merged dicts
    """
    return dict(itertools.chain(d1.items(), d2.items()))


def get_context() -> mx.context:
    """

    Returns the a list of all available gpu contexts for a given machine.
    If no gpus are available, returns [mx.cpu()].
    Use it to automatically return MxNet contexts (uses max number of gpus or cpu)

    :return: List of mxnet contexts of a gpu or [mx.cpu()] if gpu not available

    """
    context_list = []
    for gpu_number in range(16):
        try:
            _ = mx.nd.array([1, 2, 3], ctx=mx.gpu(gpu_number))
            context_list.append(mx.gpu(gpu_number))
        except mx.MXNetError:
            pass

    if len(context_list) == 0:
        context_list.append(mx.cpu())

    return context_list


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


@contextlib.contextmanager
def timing(marker):
    start_time = time.time()
    sys.stdout.flush()
    logger.info("\n========== start: %s" % marker)
    try:
        yield
    finally:
        logger.info("\n========== done (%s s) %s" % ((time.time() - start_time), marker))
        sys.stdout.flush()


class MeanSymbol(mx.metric.EvalMetric):
    """
    Metrics tracking the mean of a symbol, the index of the symbol must be passed as argument.
    """

    def __init__(self, name, symbol_index=0, output_names=None, label_names=None):
        super(MeanSymbol, self).__init__(name, output_names=output_names,
                                         label_names=label_names)
        self.symbol_index = symbol_index

    def update(self, _, preds):
        sym = preds[self.symbol_index]
        # pylint: disable=no-member
        self.sum_metric += mx.ndarray.sum(sym).asscalar()
        self.num_inst += sym.size


class AccuracyMetric(mx.metric.EvalMetric):
    """
    Metrics tracking the accuracy, the index of the discrete label must be passed as argument.
    """

    def __init__(self, name, label_index=0):
        super(AccuracyMetric, self).__init__("{}-accuracy".format(name))
        self.label_index = label_index

    def update(self, labels, preds):
        chosen = preds[1 + self.label_index].asnumpy().argmax(axis=1)
        labels_values = labels[self.label_index].asnumpy().squeeze(axis=1)
        self.sum_metric += sum((chosen == labels_values) | (labels_values == 0.0))
        self.num_inst += preds[0].size


class LogMetricCallBack(object):
    """
    Tracked the metrics specified as arguments.
    Any mxnet metric whose name contains one of the argument will be tracked.
    """

    def __init__(self, tracked_metrics, patience=None):
        """
        :param tracked_metrics: metrics to be tracked
        :param patience: if not None then if the metrics does not improve during 'patience' number
        of steps, StopIteration is raised.
        """
        self.tracked_metrics = tracked_metrics
        self.metrics = {metric: [] for metric in tracked_metrics}
        self.patience = patience

    def __call__(self, param):
        if param.eval_metric is not None:
            name_value = param.eval_metric.get_name_value()
            for metric in self.tracked_metrics:
                for name, value in name_value:
                    if metric in name and not math.isnan(value):
                        self.metrics[metric].append(value)
                        if self.patience is not None:
                            self.check_regression()

    def check_regression(self):
        """
        If no improvement happens in "patience" number of steps then StopIteration exception is
        raised. This is an ugly mechanism but it is currently the only way to support this feature
        while using module api.
        """
        _, errors = next(iter(self.metrics.items()))

        def convert_nans(e):
            if math.isnan(e):
                logger.warning("Found nan in metric")
                return 0
            else:
                return e

        errors = [convert_nans(e) for e in errors]

        if self.patience < len(errors):
            # check that the metric has improved, e.g. that all recent metrics were worse
            metric_before_patience = errors[(-1 * self.patience) - 1]

            no_improvement = all(
                [errors[-i] >= metric_before_patience for i in range(0, self.patience)]
            )
            if no_improvement:
                logger.info("No improvement detected for {} epochs compared to {} last error " \
                            "obtained: {}, stopping here".format(self.patience,
                                                                 metric_before_patience,
                                                                 errors[-1]
                                                                 ))
                raise StopIteration


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


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


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


def random_cartesian_product(sets: List,
                             num: int = 10) -> List:
    """
    Return random samples from the cartesian product of all iterators in sets.
    Returns at most as many results as unique products exist.
    Does not require materialization of the full product but is still truly random,
    wich can't be achieved with itertools.

    Example usage:
    >>> random_cartesian_product([range(2**50), ['a', 'b']], num=2)
    >>> [[558002326003088, 'a'], [367785400774751, 'a']]

    :param sets: List of iteratbles
    :param num: Number of random samples to draw
    """

    # Determine cardinality of full cartisian product
    N = np.prod([len(y) for y in sets])

    # Draw random integers without replacement
    if N > 1e6:  # avoid materialising all integers if data is large
        idxs = []
        while len(idxs) < min(num, N):
            idx_candidate = np.random.randint(N)
            if idx_candidate not in idxs:
                idxs.append(idx_candidate)
    else:
        idxs = np.random.choice(range(N), size=min(num, N), replace=False)

    out = []
    for idx in idxs:
        out.append(sample_cartesian(sets, idx, N))

    return out


def sample_cartesian(sets: List,
                     idx: int,
                     n: int = None) -> List:
    """
    Draw samples from the cartesian product of all iterables in sets.
    Each row in the cartesian product has a unique index. This function returns
    the row with index idx without materialising any of the other rows.

    For a cartesian products of lists with length l1, l2, ... lm, taking the cartesian
    product can be thought of as traversing through all lists picking one element of each
    and repeating this until all possible combinations are exhausted. The number of combinations
    is N=l1*l2*...*lm. This can make materialization of the list impracticle.
    By taking the first element from every list that leads to a new combination,
    we can define a unique enumeration of all combinations.

    :param sets: List of iteratbles
    :param idx: Index of desired row in the cartersian product
    :param n: Number of rows in the cartesian product
    """

    if n is None:
        n = np.prod([len(y) for y in sets])

    out = []  # prepare list to append elements to.
    width = n  # width of the index set in which the desired row falls.
    for item_set in sets:
        width = width/len(item_set)  # map index set onto first item_set
        bucket = int(np.floor(idx/width))  # determine index of the first item_set
        out.append(item_set[bucket])
        idx = idx - bucket*width  # restrict index to next item_set in the hierarchy (could use modulo operator here.)

    assert width == 1  # at the end of this procedure, the leaf index set should have width 1.

    return out
