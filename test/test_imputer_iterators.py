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

DataWig imputer iterator tests

"""

import os
import numpy as np
import pandas as pd
import itertools
import random
import mxnet as mx

dir_path = os.path.dirname(os.path.realpath(__file__))

from datawig.column_encoders import SequentialEncoder, CategoricalEncoder, BowEncoder
from datawig.iterators import ImputerIterDf
from test_imputer import generate_string_data_frame

random.seed(0)
np.random.seed(0)
mx.random.seed(0)

dir_path = os.path.dirname(os.path.realpath(__file__))
feature_col = "features"
label_col = "labels"
max_tokens = 100
num_labels = 10
df = generate_string_data_frame()


def get_new_iterator_df_bow():
    return ImputerIterDf(df,
                         data_columns=[BowEncoder(feature_col, max_tokens=max_tokens)],
                         label_columns=[CategoricalEncoder(label_col, max_tokens=num_labels)],
                         batch_size=2)


def get_new_iterator_df():
    return ImputerIterDf(df,
                         data_columns=[SequentialEncoder(label_col,
                                                         max_tokens=max_tokens,
                                                         seq_len=2)],
                         label_columns=[CategoricalEncoder(label_col, max_tokens=max_tokens)],
                         batch_size=2)


def test_iter_next_df():
    it = get_new_iterator_df()
    _, next_batch = next(it), next(it)
    # assert ((next_batch.data[0].asnumpy() == np.array([[31., 30.], [31., 30.]])).all())
    assert ((next_batch.label[0].asnumpy() == np.array([[9.], [9.]])).all())


def test_iter_df_bow():
    it = get_new_iterator_df_bow()
    tt = next(it)
    bow = tt.data[0].asnumpy()[0, :]
    true = it.data_columns[0].vectorizer.transform([df.loc[0,'features']]).toarray()[0]
    assert (true - bow).sum() < 1e-5


def test_iter_provide_label_or_data_df():
    it = get_new_iterator_df()
    assert (it.provide_data[0][0] == label_col)
    assert (it.provide_data[0][1] == (2, 2))
    assert (it.provide_label[0][0] == label_col)
    assert (it.provide_label[0][1] == (2, 1))


def test_iter_index_df():
    it = get_new_iterator_df()
    idx_it = list(itertools.chain(*[b.index for b in it]))
    idx_true = df.index.tolist()
    assert (idx_it == idx_true)


def test_iter_decoder_df():
    # draw skewed brands
    brands = [{feature_col: brand} for brand in
              list(map(lambda e: str(int(e)), np.random.exponential(scale=1, size=1000)))]

    brand_df = pd.DataFrame(brands)
    it = ImputerIterDf(brand_df,
                       data_columns=[SequentialEncoder(feature_col, max_tokens=10, seq_len=2)],
                       label_columns=[CategoricalEncoder(feature_col, max_tokens=100)],
                       batch_size=2)
    decoded = it.decode(next(it).label)
    np.testing.assert_array_equal(decoded[0], brand_df[feature_col].head(it.batch_size).values)


def test_iter_padding_offset():
    col = 'brand'
    df = pd.DataFrame(
        [{col: brand} for brand in
         list(map(lambda e: str(int(e)), np.random.exponential(scale=1, size=36)))]
    )
    df_train = df.sample(frac=0.5)
    it = ImputerIterDf(
        df_train,
        data_columns=[BowEncoder(col)],
        label_columns=[CategoricalEncoder(col, max_tokens=5)],
        batch_size=32
    )
    assert it.start_padding_idx == df_train.shape[0]
