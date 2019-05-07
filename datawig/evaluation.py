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

DataWig evaluation functions

"""

import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, \
    precision_recall_curve, mean_squared_error
from .utils import logger


def evaluate_and_persist_metrics(true_labels_string,
                                 true_labels_int,
                                 predictions,
                                 predictions_proba,
                                 metrics_file=None,
                                 missing_symbol=None,
                                 numerical_labels={},
                                 numerical_predictions={}):
    '''
    Compute and save metrics in metrics_file
    :param true_labels_string: dict with true labels e.g. {'color':["black","white"]}
    :param true_labels_int: dict with true label indices e.g. {'color':[0,1]}
    :param predictions: dict with predicted labels e.g. {'color':["black","white"]}
    :param predictions_proba: dict with class likelihoods e.g. {'color':[0.1,0.9]}
    :param true_labels_int: dict with predicted class index e.g. {'color':[1,0]}
    :param metrics_file: path to *.json file to store metrics
    :param missing_symbol: dict with missing symbol (will be excluded from metrics) e.g.
                        {'color': "MISSING"}
    :param numerical_labels: dict with true numerical outputs
    :param numerical_predictions: dict with predicted numerical outputs
    :return:
    '''

    if missing_symbol is None:
        missing_symbol = {k: "MISSING" for k in true_labels_string.keys()}

    if len(true_labels_string) > 0:
        # concatenate all categorical predictions
        dfs = []
        for att in true_labels_string.keys():
            true_and_predicted = [(true, pred) for true, pred in zip(true_labels_string[att], predictions[att]) if
                                  true != missing_symbol[att]]
            assert len(true_and_predicted) > 0, "No valid ground truth data for label {}".format(att)
            truth, predicted = zip(*true_and_predicted)
            logger.debug(
                "Keeping {}/{} not-missing values for evaluation of {}".format(len(truth), len(true_labels_string[att]),
                                                                               att))
            dfs.append(pd.DataFrame({'attribute': att, 'true_value': truth, 'predicted_value': predicted}))

        df = pd.concat(dfs)

        metrics = evaluate_model_outputs(df)

        logger.debug("average classification metrics:")
        for label, att_metrics in metrics.items():
            logger.debug(
                "label: {} - {}".format(label, list(filter(lambda x: x[0].startswith("avg"), att_metrics.items()))))

        logger.debug("weighted average classification metrics:")
        for label, att_metrics in metrics.items():
            logger.debug(
                "label: {} - {}".format(label,
                                        list(filter(lambda x: x[0].startswith("weighted"), att_metrics.items()))))

        for att in true_labels_string.keys():
            metrics[att]['precision_recall_curves'] = {}
            for label in range(1, predictions_proba[att].shape[-1]):
                true_labels = (true_labels_int[att] == label).nonzero()[0]
                if len(true_labels) > 0:
                    true_label = true_labels_string[att][true_labels[0]]
                    y_true = (true_labels_int[att] == label) * 1.0
                    y_score = predictions_proba[att][:, label]
                    prec, rec, thresholds = precision_recall_curve(y_true, y_score)
                    metrics[att]['precision_recall_curves'][true_label] = {'precision': prec,
                                                                           'recall': rec,
                                                                           'thresholds': thresholds}
                    threshold_idx = (prec > .95).nonzero()[0][0] - 1
                    logger.debug(
                        "Attribute {}, Label: {}\tReaching {} precision / {} recall at threshold {}".format(
                            att, true_label, prec[threshold_idx], rec[threshold_idx], thresholds[threshold_idx]))
    else:
        metrics = {}

    for col_name in numerical_labels.keys():
        metrics[col_name] = 1.0 * mean_squared_error(numerical_labels[col_name], numerical_predictions[col_name])

    if metrics_file:
        import copy
        serialize_metrics = copy.deepcopy(metrics)
        # transform precision_recall_curves to json serializable lists
        for att in true_labels_string.keys():
            for label in metrics[att]['precision_recall_curves'].keys():
                serialize_metrics[att]['precision_recall_curves'][label] = \
                    {
                        'precision': metrics[att]['precision_recall_curves'][label]['precision'].tolist(),
                        'recall': metrics[att]['precision_recall_curves'][label]['recall'].tolist(),
                        'thresholds': metrics[att]['precision_recall_curves'][label]['thresholds'].tolist()
                    }
        logger.debug("save metrics in {}".format(metrics_file))
        with open(metrics_file, "w") as fp:
            json.dump(serialize_metrics, fp)

    return metrics


def evaluate_model_outputs(
        df,
        attribute_column_name='attribute',
        true_label_column_name='true_value',
        predicted_label_column_name='predicted_value'):
    '''

    :param df: a dataframe with attribute | true_value | predicted_value
    :return: precision/recall/f1/accuracy for each attribute (class frequency weighted) and for each column-label combination
    '''

    groups = df.groupby(attribute_column_name)

    model_metrics = {}

    for group, group_df in groups:
        true = group_df[true_label_column_name]
        predicted = group_df[predicted_label_column_name]

        model_metrics[group] = evaluate_model_outputs_single_attribute(true, predicted)

    return model_metrics


def evaluate_model_outputs_single_attribute(true, predicted, topMisclassifications=100):
    true = true.astype(str)
    predicted = predicted.astype(str)

    labels = true.value_counts()

    model_metrics = dict()

    model_metrics['class_counts'] = [(a, int(b)) for a, b in labels.iteritems()]

    # computes statistics not weighted by class frequency
    model_metrics['avg_precision'] = precision_score(true, predicted, average='macro')
    model_metrics['avg_recall'] = recall_score(true, predicted, average='macro')
    model_metrics['avg_f1'] = f1_score(true, predicted, average='macro')
    model_metrics['avg_accuracy'] = accuracy_score(true, predicted)

    # computes statistics weighted by class frequency
    model_metrics['weighted_precision'] = precision_score(true, predicted, average='weighted')
    model_metrics['weighted_recall'] = recall_score(true, predicted, average='weighted')
    model_metrics['weighted_f1'] = f1_score(true, predicted, average='weighted')
    # todo sample_weight seems missing
    model_metrics['weighted_accuracy'] = accuracy_score(true, predicted)

    # single class metrics
    model_metrics['class_precision'] = precision_score(true, predicted, average=None,
                                                       labels=labels.index.tolist()).tolist()
    model_metrics['class_recall'] = recall_score(true, predicted, average=None, labels=labels.index.tolist()).tolist()
    model_metrics['class_f1'] = f1_score(true, predicted, average=None, labels=labels.index.tolist()).tolist()
    model_metrics['class_accuracy'] = accuracy_score(true, predicted).tolist()
    model_metrics['num_classes'] = len(model_metrics['class_counts'])
    model_metrics['num_applicable_rows'] = int(sum(count for (class_name, count) in model_metrics['class_counts']))

    true_name = "true"
    pred_name = "pred"
    groups = pd.DataFrame(list(zip(true, predicted)), columns=[true_name, pred_name]).groupby(true_name)
    model_metrics['confusion_matrix'] = []
    for label in labels.index.tolist():
        confusion_matrix_series = groups.get_group(label)[pred_name].value_counts()[:topMisclassifications]
        confusion_matrix = list(zip(confusion_matrix_series.index.tolist(), map(int, confusion_matrix_series.tolist())))
        model_metrics['confusion_matrix'].append((label, confusion_matrix))

    return model_metrics
