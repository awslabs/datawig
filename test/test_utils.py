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

DataWig utils tests

"""

import numpy as np
import pandas as pd

from datawig.utils import random_split, normalize_dataframe, merge_dicts


def test_random_split():
    df = pd.DataFrame([{'a': 1}, {'a': 2}])
    train_df, test_df = random_split(df, split_ratios=[.5, .5], seed=10)
    assert all(train_df.values.flatten() == np.array([1]))
    assert all(test_df.values.flatten() == np.array([2]))


def test_normalize_dataframe():
    assert (normalize_dataframe(pd.DataFrame({'a': ['     Aƒa    ', 2]}))['a'].values.tolist() == [
        'aa', '2'])


def test_merge_dicts():
    d1 = {'a': 1}
    d2 = {'b': 2}
    merged = merge_dicts(d1, d2)
    assert merged == {'a': 1, 'b': 2}
