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
import pytest


def test_fit_predict():
    df = pd.read_csv(os.path.join('examples', 'mae_train_dataset.csv'))
    imputer = SimpleImputer(['text', 'title'], 'finish').fit(df)
    predictions = imputer.predict(df)

    assert predictions.shape[0] == df.shape[0]


def test_fit_hpo():
    df = pd.read_csv(os.path.join('examples', 'mae_train_dataset.csv'))
    imputer = SimpleImputer(['text', 'title'], 'finish')

    grid = {
        'text': {'max_features': [2**12], 'analyzer': ['char']},
        'title': {'analyzer': ['char']},
        'estimator': {'alpha': [1e-1]}
    }
    imputer.fit_hpo(df, grid)

    assert imputer.model.named_steps['transformers'].named_transformers_['text'].max_features == 2**12
    assert imputer.model.named_steps['transformers'].named_transformers_['text'].analyzer == 'char'
    assert imputer.model.named_steps['transformers'].named_transformers_['title'].analyzer == 'char'
    assert imputer.model.named_steps['estimator'].base_estimator.alpha == 1e-1


def test_explain():
    df = pd.read_csv(os.path.join('examples', 'mae_train_dataset.csv'))
    imputer = SimpleImputer(['text', 'title'], 'finish')

    imputer.fit_hpo(df, {})
    explanations = imputer.explain('black', k=3)

    assert 'text' in explanations
    assert 'title' in explanations
    assert explanations['text'] == ['black', 'the', 'this']
    assert explanations['title'] == ['black', 'nikon', 'curtain']


def test_explain_instance():
    # df = pd.read_csv(os.path.join('examples', 'mae_train_dataset.csv'))
    # imputer = SimpleImputer(['text', 'title'], 'finish')
    #
    # imputer.fit_hpo(df, {})
    # explanations = imputer.explain_instance(df.iloc[19], k=3)
    #
    # assert 'text' in explanations
    # assert 'title' in explanations
    # assert explanations['text'] == ['black', 'the', 'this']
    # assert explanations['title'] == ['black', 'nikon', 'curtain']
    pytest.fail("PENDING")


# def test_save_load(test_dir):
#     output_dir = os.path.join(test_dir)
#     df = pd.read_csv(os.path.join('examples', 'mae_train_dataset.csv'))
#     imputer = SimpleImputer(['text', 'title'], 'finish').fit(df)
#     imputer.save(output_dir)
#     loaded_imputer = SimpleImputer.load(output_dir)

