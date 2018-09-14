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

DataWig ColumnEncoder tests

"""

import pytest
import os
import random
import shutil
import pandas as pd
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

from datawig import column_encoders
from test.test_imputer import create_test_image

random.seed(1)
np.random.seed(42)

df = pd.DataFrame({'features': ['xwcxG pQldP Cel0n 5LaWO 2cjTu',
                                '2cjTu YizDY u1aEa Cel0n SntTK',
                                '2cjTu YizDY u1aEa Cel0n SntTK'],
                   'labels': ['xwcxG', 'SntTK', 'SntTK']})

categorical_encoder = column_encoders.CategoricalEncoder(['labels'], max_tokens=3).fit(df)
sequential_encoder = column_encoders.SequentialEncoder(['features'],
                                                       max_tokens=50, seq_len=3).fit(df)


# CategoricalEncoder Tests
def test_categorical_encoder_unfitted_fail():
    unfitted_categorical_encoder = column_encoders.CategoricalEncoder(["col_1"])
    assert unfitted_categorical_encoder.is_fitted() == False
    with pytest.raises(column_encoders.NotFittedError):
        unfitted_categorical_encoder.transform(pd.DataFrame({"col_1": ['a', 'b']}))


def test_fit_categorical_encoder():
    assert categorical_encoder.is_fitted() == True
    assert (categorical_encoder.token_to_idx == {'SntTK': 1, 'xwcxG': 2})
    assert (categorical_encoder.idx_to_token == {1: 'SntTK', 2: 'xwcxG'})


def test_categorical_encoder_transform():
    assert categorical_encoder.transform(df).flatten()[0] == 2.


def test_categorical_encoder_transform_missing_token():
    assert (categorical_encoder.transform(pd.DataFrame({'labels': ['foobar']})).flatten()[0] == 0)


def test_categorical_encoder_max_token():
    categorical_encoder = column_encoders.CategoricalEncoder(['labels'], max_tokens=1e4).fit(df)
    assert (categorical_encoder.max_tokens == 2)


def test_categorical_encoder_decode_token():
    assert (categorical_encoder.decode_token(1) == 'SntTK')


def test_categorical_encoder_decode_missing_token():
    assert (categorical_encoder.decode_token(0) == 'MISSING')


def test_categorical_encoder_decode():
    assert (categorical_encoder.decode(pd.Series([1])).values[0] == 'SntTK')


def test_categorical_encoder_decode_missing():
    assert (categorical_encoder.decode(pd.Series([0])).values[0] == 'MISSING')


def test_categorical_encoder_non_negative_embedding_indices():
    assert all(categorical_encoder.transform(df).flatten() >= 0)


# SequentialEncoder Tests
def test_sequential_encoder_unfitted_fail():
    unfitted_sequential_encoder = column_encoders.SequentialEncoder(["col_1"])
    assert unfitted_sequential_encoder.is_fitted() == False
    with pytest.raises(column_encoders.NotFittedError):
        unfitted_sequential_encoder.transform(pd.DataFrame({'brand': ['ab']}))


def test_fit_sequential_encoder():
    sequential_encoder_fewer_tokens = column_encoders.SequentialEncoder(['features'],
                                                                        max_tokens=5,
                                                                        seq_len=3).fit(df)
    assert (set(sequential_encoder_fewer_tokens.token_to_idx.keys()) == {'u', 'a', 'n', 'T', ' '})


def test_sequential_encoder_transform():
    encoded = pd.Series([vec.tolist() for vec in sequential_encoder.transform(df)])
    true_decoded = df['features'].apply(lambda x: x[:sequential_encoder.output_dim])
    assert all(sequential_encoder.decode(encoded) == true_decoded)


def test_sequential_encoder_transform_missing_token():
    assert (sequential_encoder.transform(pd.DataFrame({'features': ['!~']}))[0].tolist() == [0, 0,
                                                                                             0])


def test_sequential_encoder_max_token():
    sequential_encoder_short = column_encoders.SequentialEncoder("features", max_tokens=1e4,
                                                                 seq_len=2)
    sequential_encoder_short.fit(df)
    assert sequential_encoder.is_fitted() == True
    assert (sequential_encoder_short.max_tokens == 32)


def test_sequential_encoder_non_negative_embedding_indices():
    assert all(sequential_encoder.transform(df).flatten() >= 0)


def test_bow_encoder():
    bow_encoder = column_encoders.BowEncoder("features", max_tokens=5)
    assert bow_encoder.is_fitted() == True
    bow = bow_encoder.transform(df)[0].toarray()[0]
    true = np.array([0.615587, -0.3077935, -0.3077935, -0.41039133, 0.51298916])
    assert true == pytest.approx(bow, 1e-4)


def test_bow_encoder_multicol():
    bow_encoder = column_encoders.BowEncoder(["item_name", "product_description"], max_tokens=5)
    data = pd.DataFrame({'item_name': ['bla'], 'product_description': ['fasl']})
    bow = bow_encoder.transform(data)[0].toarray()[0]
    true = np.array([0.27500955, -0.82502865, -0.1833397, 0., -0.45834925])
    assert true == pytest.approx(bow, 1e-4)
    data_strings = ['item_name bla product_description fasl ']
    assert true == pytest.approx(bow_encoder.vectorizer.transform(data_strings).toarray()[0])


def test_categorical_encoder_numeric():
    df = pd.DataFrame({'brand': [1, 2, 3]})
    try:
        column_encoders.CategoricalEncoder("brand").fit(df)
    except TypeError:
        pytest.fail("fitting categorical encoder on integers should not fail")


def test_categorical_encoder_numeric_transform():
    df = pd.DataFrame({'brand': [1, 2, 3, 1, 2, 1, np.nan, None]})
    col_enc = column_encoders.CategoricalEncoder("brand").fit(df)
    assert np.array_equal(col_enc.transform(df), np.array([[1], [2], [3], [1], [2], [1], [0], [0]]))


def test_categorical_encoder_numeric_nan():
    df = pd.DataFrame({'brand': [1, 2, 3, None]})
    try:
        column_encoders.CategoricalEncoder("brand").fit(df)
    except TypeError:
        pytest.fail("fitting categorical encoder on integers with nulls should not fail")

def test_column_encoder_no_list_input_column():
     column_encoder = column_encoders.ColumnEncoder("0")
     assert column_encoder.input_columns == ['0']
     assert column_encoder.output_column == '0'
     with pytest.raises(ValueError):
        column_encoders.ColumnEncoder(0)
     with pytest.raises(ValueError):
         column_encoders.ColumnEncoder([0])


def test_numeric_encoder():
    df = pd.DataFrame({'a': [1, 2, 3, np.nan, None], 'b': [.1, -.1, np.nan, None, 10.5]})
    unfitted_numerical_encoder = column_encoders.NumericalEncoder(["a", 'b'], normalize=False)
    assert unfitted_numerical_encoder.is_fitted() == True
    fitted_unnormalized_numerical_encoder = unfitted_numerical_encoder.fit(df)
    df_unnormalized = fitted_unnormalized_numerical_encoder.transform(df.copy())

    assert np.array_equal(df_unnormalized, np.array([[1., 0.1],
                                                     [2., -0.1],
                                                     [3., 3.5],
                                                     [2.0, 3.5],
                                                     [2.0, 10.5]], dtype=np.float32))
    df_nans = pd.DataFrame({'a': [None], 'b': [np.nan]})
    df_unnormalized_nans = fitted_unnormalized_numerical_encoder.transform(df_nans.copy())
    assert np.array_equal(df_unnormalized_nans, np.array([[2., 3.5]], dtype=np.float32))

    normalized_numerical_encoder = column_encoders.NumericalEncoder(["a", 'b'], normalize=True)
    assert normalized_numerical_encoder.is_fitted() == False
    normalized_numerical_encoder_fitted = normalized_numerical_encoder.fit(df)
    df_normalized = normalized_numerical_encoder_fitted.transform(df)
    df_normalized_nans = normalized_numerical_encoder_fitted.transform(df_nans)

    assert normalized_numerical_encoder.is_fitted() == True
    assert np.array_equal(df_normalized, np.array([[-1.58113885, -0.88666826],
                                                   [0., -0.93882525],
                                                   [1.58113885, 0.],
                                                   [0., 0.],
                                                   [0., 1.82549345]], dtype=np.float32))


def test_image_encoder():

    img_path = os.path.join(dir_path, "resources", "test_images")
    os.makedirs(img_path, exist_ok=True)

    colors = ['red', 'green']
    img_filenames = ['not-existent.jpg']
    for color in colors:
        filename = os.path.join(img_path, color + ".png")
        create_test_image(filename, color)
        img_filenames.append(filename)

    df = pd.DataFrame({"test_uris": img_filenames})

    untransformed_image_encoder = column_encoders.ImageEncoder(['test_uris'])
    tensor = untransformed_image_encoder.transform(df)
    assert tensor[:,:,1,1] == pytest.approx(np.array([[ 0.,  0., 0.],
       [ 0.53643285, -2.03571422, -1.80444444],
       [-2.11790397,  0.67787113, -1.80444444]]), 1e-5)

    shutil.rmtree(img_path)
