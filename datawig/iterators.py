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

DataWig Iterators:
used for feeding data in a pandas DataFrame into an MxNet Imputer Module

"""
from typing import List, Any
import mxnet as mx
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from .column_encoders import ColumnEncoder, NumericalEncoder
from .utils import logger
INSTANCE_WEIGHT_COLUMN = '__empirical_risk_instance_weight__'


class ImputerIter(mx.io.DataIter):
    """

    Constructs an MxNet Iterator for a datawig.imputation.Imputer given a table in csv format

    :param data_columns: list of featurizers, see datawig.column_featurizer
    :param label_columns: list of featurizers
    :param batch_size: size of minibatches

    """

    def __init__(self,
                 data_columns: List[ColumnEncoder],
                 label_columns: List[ColumnEncoder],
                 batch_size: int = 512) -> None:

        mx.io.DataIter.__init__(self, batch_size)

        self.data_columns = data_columns
        self.label_columns = label_columns
        self.cur_batch = 0

        self._provide_data = None
        self._provide_label = None
        self.df_iterator = None
        self.indices = []
        self.start_padding_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        """

        Returning _provide_data
        :return:

        """
        return self._provide_data

    @property
    def provide_label(self):
        """

        Returning _provide_label
        :return:

        """
        return self._provide_label

    def decode(self, mxnet_label_predictions: Any) -> Any:
        """

        Takes a list of mxnet label predictions, returns decoded symbols,
        Decoding is done with respective ColumnEncoder

        :param mxnet_label_predictions:
        :return: Decoded labels
        """
        return [col.decode(pd.Series(pred.asnumpy().flatten())).tolist() for col, pred in
                zip(self.label_columns, mxnet_label_predictions)]

    def mxnet_iterator_from_df(self, data_frame: pd.DataFrame) -> mx.io.NDArrayIter:
        """

        Takes pandas DataFrame and returns set an MxNet iterator

        :return: MxNet iterator
        """
        n_samples = len(data_frame)
        # transform data into mxnet nd arrays
        data = {}
        for col_enc in self.data_columns:
            data_array_numpy = col_enc.transform(data_frame)
            data[col_enc.output_column] = mx.nd.array(data_array_numpy[:n_samples, :])
            logger.debug("Data Encoding - Encoded {} rows of column \
                        {} with {} into \
                        {} of shape {} \
                        and then into shape {}".format(len(data_frame),
                        ",".join(col_enc.input_columns), col_enc.__class__, type(data_array_numpy), data_array_numpy.shape, data[col_enc.output_column].shape))

        # transform labels into mxnet nd arrays
        labels = {}
        for col_enc in self.label_columns:
            assert len(col_enc.input_columns) == 1, "Number of encoder input columns for labels  \
                                                    must be 1, was {}".format(len(col_enc.input_columns))

            if col_enc.input_columns[0] in data_frame.columns:
                labels_array_numpy = col_enc.transform(data_frame).astype(np.float64)
                labels[col_enc.output_column] = mx.nd.array(labels_array_numpy[:n_samples, :])
                logger.debug("Label Encoding - Encoded {} rows of column \
                            {} with {} into \
                            {} of shape {} and \
                            then into shape {}".format(len(data_frame), col_enc.input_columns[0], col_enc.__class__, type(labels_array_numpy), labels_array_numpy.shape, labels[col_enc.output_column].shape))
            else:
                labels[col_enc.input_columns[0]] = mx.nd.zeros((n_samples, 1))
                logger.debug("Could not find column {} in DataFrame, \
                             setting {} labels to missing".format(col_enc.input_columns[0], n_samples))

        # transform label weights to mxnet nd array
        assert len(labels.keys()) == 1  # make sure we only have one output label

        # numerical label encoder can't handle class weights
        if not isinstance(self.label_columns[0], NumericalEncoder):

            # add instance weights, set to all ones if no such column is in the data.
            if INSTANCE_WEIGHT_COLUMN in data_frame.columns:
                data[INSTANCE_WEIGHT_COLUMN] = mx.nd.array(np.expand_dims(
                    data_frame[INSTANCE_WEIGHT_COLUMN], 1))
            else:
                data[INSTANCE_WEIGHT_COLUMN] = mx.nd.array(np.ones([n_samples, 1]))

        # mxnet requires to use last_batch_handle='discard' for sparse data
        # if there are not enough data points for a batch, we cannot construct an iterator
        return mx.io.NDArrayIter(data, labels, batch_size=self.batch_size,
                                 last_batch_handle='discard')

    def _n_rows_padding(self, data_frame: pd.DataFrame) -> int:
        """
        Returns the number of rows needed to make the number of rows in `data_frame`
        divisable by self.batch_size without remainder.

        :param data_frame: pandas.DataFrame
        :return: int, number of rows to pad

        """
        n_test_samples = data_frame.shape[0]
        n_rows = int(self.batch_size - n_test_samples % self.batch_size)

        pad = 0
        if n_rows != self.batch_size:
            pad = n_rows

        return pad

    def reset(self):
        """

        Resets Iterator

        """
        self.cur_batch = 0
        self.df_iterator.reset()

    def next(self) -> mx.io.DataBatch:
        """
        Returns next batch of data

        :return:
        """

        # get the next batch from the underlying mxnet ndarrayiter
        next_batch = next(self.df_iterator)

        # and add indices from original dataframe, if data didn't come from an mxnet iterator
        start_batch = self.cur_batch * self.batch_size
        next_batch.index = self.indices[start_batch:start_batch + self.batch_size]

        self.cur_batch += 1

        return next_batch


class ImputerIterDf(ImputerIter):
    """

    Constructs an MxNet Iterator for a datawig.imputation.Imputer given a pandas dataframe

    :param data_frame: pandas dataframe
    :param data_columns: list of featurizers, see datawig.column_featurizer
    :param label_columns: list of featurizers [CategoricalFeaturizer('field_name_1')]
    :param batch_size: size of minibatches

    """

    def __init__(self,
                 data_frame: pd.DataFrame,
                 data_columns: List[ColumnEncoder],
                 label_columns: List[ColumnEncoder],
                 batch_size: int = 512) -> None:
        super(ImputerIterDf, self).__init__(data_columns, label_columns, batch_size)

        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        # fill string nan with empty string, numerical nan with np.nan
        numerical_columns = [c for c in data_frame.columns if is_numeric_dtype(data_frame[c])]
        string_columns = list(set(data_frame.columns) - set(numerical_columns))
        data_frame = data_frame.fillna(value={x: "" for x in string_columns})
        data_frame = data_frame.fillna(value={x: np.nan for x in numerical_columns})

        self.indices = data_frame.index.tolist()
        data_frame = data_frame.reset_index(drop=True)

        # custom padding for having to discard the last batch in mxnet for sparse data
        padding_n_rows = self._n_rows_padding(data_frame)
        self.start_padding_idx = int(data_frame.index.max() + 1)
        for idx in range(self.start_padding_idx, self.start_padding_idx + padding_n_rows):
            data_frame.loc[idx, :] = data_frame.loc[self.start_padding_idx - 1, :]

        for column_encoder in self.data_columns + self.label_columns:
            # ensure that column encoder is fitted to data before applying it
            if not column_encoder.is_fitted():
                column_encoder.fit(data_frame)

        self.df_iterator = self.mxnet_iterator_from_df(data_frame)
        self.df_iterator.reset()
        self._provide_data = self.df_iterator.provide_data
        self._provide_label = self.df_iterator.provide_label
