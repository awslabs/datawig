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

DataWig tests for explaining predictions

"""

import os
import datawig
from datawig.utils import random_split
from datawig.utils import logger
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
from datawig import Imputer
logger.setLevel("DEBUG")


def test_explain_method_synthetic(test_dir):

    # Generate simulated data for testing explain method
    # Predict output column with entries in ['foo', 'bar'] from two columns, one
    # categorical in ['foo', 'dummy'], one text in ['text_foo_text', 'text_dummy_text'].
    # the output column is deterministically 'foo', if 'foo' occurs anywhere in any input column.
    N = 100
    cat_in_col = ['foo' if r > (1 / 2) else 'dummy' for r in np.random.rand(N)]
    text_in_col = ['fff' if r > (1 / 2) else 'ddd' for r in np.random.rand(N)]
    hash_in_col = ['h' for r in range(N)]
    cat_out_col = ['foo' if 'f' in input[0] + input[1] else 'bar' for input in zip(cat_in_col, text_in_col)]

    df = pd.DataFrame()
    df['in_cat'] = cat_in_col
    df['in_text'] = text_in_col
    df['in_text_hash'] = hash_in_col
    df['out_cat'] = cat_out_col

    # Specify encoders and featurizers #
    data_encoder_cols = [datawig.column_encoders.TfIdfEncoder('in_text', tokens="chars"),
                         datawig.column_encoders.CategoricalEncoder('in_cat', max_tokens=1e1),
                         datawig.column_encoders.BowEncoder('in_text_hash', tokens="chars")]
    data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('in_text'),
                            datawig.mxnet_input_symbols.EmbeddingFeaturizer('in_cat'),
                            datawig.mxnet_input_symbols.BowFeaturizer('in_text_hash')]

    label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('out_cat')]

    # Specify model
    imputer = datawig.Imputer(
       data_featurizers=data_featurizer_cols,
       label_encoders=label_encoder_cols,
       data_encoders=data_encoder_cols,
       output_path=os.path.join(test_dir, "tmp", "explanation_tests")
       )

    # Train
    tr, te = random_split(df.sample(90), [.8, .2])
    imputer.fit(train_df=tr, test_df=te, num_epochs=20, learning_rate = 1e-2)
    predictions = imputer.predict(te)

    # Evaluate
    assert precision_score(predictions.out_cat, predictions.out_cat_imputed, average='weighted') > .99

    # assert item explanation, iterate over some inputs
    for i in np.random.choice(N, 10):
        # any instance label needs to be explained by at least on appropriate input column
        instance_explained_by_appropriate_feature = False

        explanation = imputer.explain_instance(df.iloc[i])
        top_label = explanation['explained_label']

        if top_label == 'bar':
            assert (explanation['in_text'][0][0] == 'd' and explanation['in_cat'][0][0] == 'dummy')
        elif top_label == 'foo':
            assert (explanation['in_text'][0][0] == 'f' or explanation['in_cat'][0][0] == 'foo')

    # assert class explanations
    assert np.all(['f' in token for token, weight in imputer.explain('foo')['in_text']][:3])
    assert ['f' in token for token, weight in imputer.explain('foo')['in_cat']][0]

    # test serialisation to disk
    imputer.save()
    imputer_from_disk = Imputer.load(imputer.output_path)
    assert np.all(['f' in token for token, weight in imputer_from_disk.explain('foo')['in_text']][:3])
    