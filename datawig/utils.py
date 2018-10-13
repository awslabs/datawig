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
import os
import random
import sys
import time
from typing import Any, List, Tuple

import mxnet as mx
import numpy as np
import pandas as pd

mx.random.seed(1)
random.seed(1)
np.random.seed(42)

log_formatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s")
logger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(log_formatter)
logger.addHandler(consoleHandler)

logger.setLevel("INFO")

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
        split_ratios = [.8,.2]
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
