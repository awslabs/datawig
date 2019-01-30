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

from datawig._hpo import _HPO
from datawig.simple_imputer import SimpleImputer


def test_single_hpo(test_dir, data_frame):
    feature_col, label_col = "feature", "label"

    df = data_frame(feature_col=feature_col,
                    label_col=label_col)

    imputer = SimpleImputer(
        input_columns=[col for col in df.columns if col != label_col],
        output_column=label_col,
        output_path=test_dir
    )

    hps = dict()
    hps[feature_col] = {'max_tokens': [1024]}
    hps['global'] = {}
    hps['global']['num_epochs'] = [10]
    hps['string'] = {}
    hps['categorical'] = {}
    hps['numeric'] = {}

    hpo = _HPO()
    hpo.tune(
        train_df=df,
        hps=hps,
        simple_imputer=imputer
    )

    assert hpo.results.shape[0] == 1
    assert hpo.results[feature_col+':max_tokens'].values[0] == 1024
    assert hpo.results['global:num_epochs'].values[0] == 10
