# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from datawig.simple_imputer import ScikitImputer as SimpleImputer
import os
import pandas as pd


def test_init(test_dir):
    # output_dir = os.path.join(test_dir)
    df = pd.read_csv(os.path.join('..', 'examples', 'mae_train_dataset.csv'))
    imputer = SimpleImputer(['text', 'title'], 'finish').fit(df)
    imputer.save(test_dir)
    imputer2 = SimpleImputer.load(test_dir)
    assert (imputer.predict(df).values == imputer2.predict(df).values).mean() == 1.0
