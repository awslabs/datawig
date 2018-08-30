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

DataWig evaluation tests

"""

import warnings
import pandas as pd
from datawig.evaluation import evaluate_model_outputs, evaluate_model_outputs_single_attribute

warnings.filterwarnings("ignore")

def test_evaluation_single():
    str_values = pd.Series(['foo', 'bar'])
    int_values = pd.Series([1, 2])
    float_values = pd.Series([1., 2.])
    metrics_str = evaluate_model_outputs_single_attribute(str_values, str_values)
    metrics_int = evaluate_model_outputs_single_attribute(int_values, int_values)
    metrics_float = evaluate_model_outputs_single_attribute(float_values, float_values)

    assert metrics_str['avg_accuracy'] == 1.0
    assert metrics_int['avg_accuracy'] == 1.0
    assert metrics_float['avg_accuracy'] == 1.0


def test_evaluation():
    df = pd.DataFrame(
        [
            ('color', 'black', 'black'),
            ('color', 'white', 'white'),
            ('color', 'black', 'black')
        ], columns=['attribute', 'true_value', 'predicted_value']
    )

    correct_df = {'color': {'avg_accuracy': 1.0,
                            'avg_f1': 1.0,
                            'avg_precision': 1.0,
                            'avg_recall': 1.0,
                            'weighted_accuracy': 1.0,
                            'weighted_f1': 1.0,
                            'weighted_precision': 1.0,
                            'weighted_recall': 1.0,
                            'class_accuracy': 1.0,
                            'class_counts': [('black', 2), ('white', 1)],
                            'class_f1': [1.0, 1.0],
                            'class_precision': [1.0, 1.0],
                            'class_recall': [1.0, 1.0],
                            'confusion_matrix': [('black', [('black', 2)]), ('white', [('white', 1)])],
                            'num_applicable_rows': 3,
                            'num_classes': 2}}

    evaluation_df = evaluate_model_outputs(df)

    assert (evaluation_df == correct_df)

    df = pd.DataFrame(
        [
            ('color', 'black', 'black'),
            ('color', 'white', 'black'),
            ('color', 'white', 'black')
        ], columns=['attribute', 'true_value', 'predicted_value']
    )

    wrong_df = {'color': {'avg_accuracy': 1.0 / 3.0,
                          'avg_f1': 1.0 / 4.0,
                          'avg_precision': 1.0 / 6.0,
                          'avg_recall': 1.0 / 2.0,
                          'weighted_accuracy': 1.0 / 3.0,
                          'weighted_f1': 1.0 / 6.0,
                          'weighted_precision': 1.0 / 9.0,
                          'weighted_recall': 1.0 / 3.0,
                          'class_accuracy': 1.0 / 3.0,
                          'class_counts': [('white', 2), ('black', 1)],
                          'class_f1': [0.0, 0.5],
                          'class_precision': [0.0, 1.0 / 3.0],
                          'class_recall': [0.0, 1.0],
                          'confusion_matrix': [('white', [('black', 2)]), ('black', [('black', 1)])],
                          'num_applicable_rows': 3,
                          'num_classes': 2}}

    evaluation_df = evaluate_model_outputs(df)

    assert (evaluation_df == wrong_df)
