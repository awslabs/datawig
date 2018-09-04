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

Methods for model calibration via temperature scaling,
applied as post-processing step.

"""


import numpy as np
from scipy.optimize import minimize
from .utils import softmax


def compute_ece(scores: np.ndarray,
                labels: np.ndarray,
                lbda: float = 1,
                step: float = .05) -> float:
    """

    :param scores: Probabilities or logit scores of dimension samples x classes.
    :param labels: True labels as corresponding column indices
    :param lbda: scaling parameter, lbda = 1 amounts to no rescaling.
    :param step: histogram bin width. Can be tuned.

    :return:

    """

    # compute probabilities applying tempered softmax
    probas = calibrate(scores, lbda)

    # probabilities for the most likely labels
    top_probas = np.max(probas, axis=1)

    # top prediction is independent of temperature scaling or logits/probas
    predictions = np.argmax(probas, 1)

    bin_means = np.arange(step / 2, 1, step)

    ece = 0

    # iterate over bins and compare confidence and precision
    for bin_lower, bin_upper in [(mean - step / 2, mean + step / 2) for mean in bin_means]:

        # select bin entries
        bin_mask = (top_probas >= bin_lower) & (top_probas < bin_upper)
        if np.any(bin_mask):
            in_bin_confidence = np.mean(top_probas[bin_mask])
            in_bin_precision = np.mean(labels[bin_mask] == predictions[bin_mask])
            ece += np.abs(in_bin_confidence - in_bin_precision) * np.mean(bin_mask)

    return ece


def ece_loss(lbda: float, *args) -> float:
    """

    :param lbda: Scaling parameter
    wrapper around compute_ece() to be called during optimisation

    """

    scores, labels = args

    return compute_ece(scores, labels, lbda=lbda)


def logits_from_probas(probas: np.ndarray, force: bool=False) -> np.ndarray:
    """
    Returns logits for a vector of class probabilities. This is not a unique transformation.
    If the input is not a probability, no transformation is made by default.

    :param probas: Probabilities of dimension samples x classes.
    :param force: True forces rescaling, False only rescales if the values look like probabilities.
    """

    # check whether rows of input are probabilities
    if (force is True) or (np.all(np.sum(probas, 1) - 1 < 1e-12) and np.all(probas <= 1)):
        return np.log(probas)
    else:
        return probas


def probas_from_logits(scores: np.ndarray, lbda: float=1, force: bool=False) -> np.ndarray:
    """
    Returns probabilitiess for a vector of class logits.
    If the input is a probability, no transformation is made.

    :param scores: Logits of dimension samples x classes.
    :param lbda: parameter for temperature scaling
    :param force: True forces rescaling, False only rescales if the values don't look like probabilities.
    """

    # check whether rows of input are probabilities
    if np.all(np.sum(scores, 1) - 1 < 1e-2) and np.all(scores <= 1) and force is False:
        return scores
    else:
        return np.array([softmax(lbda * row) for row in scores])


def calibrate(scores: np.ndarray, lbda: float) -> np.ndarray:
    """
    Apply temperature scaling

    :param scores: Probabilities of dimension samples x classes. Do not pass logits.
    :param lbda: Parameter for temperature scaling.
    :return: Calibrated array of probabilities of dimensions samples x classes.
    """

    logits = logits_from_probas(scores, force=True)

    return np.array([softmax(lbda * row) for row in logits])


def reliability(scores: np.ndarray, labels: np.ndarray, step: float=.05) -> tuple:
    """
    Compute tuples for reliability plots.

    :param scores: Probabilities or logits of dimension samples x classes.
    :param labels: True labels as corresponding column indices
    :param step: histogram bin width. Can be tuned.
    :return: tuple containing mean of bins and the precision in each bin.
    """

    # transform scores to probabilities if applicable
    probas = probas_from_logits(scores)

    # probabilities for the most likely labels
    top_probas = np.max(probas, axis=1)

    predictions = np.argmax(probas, 1)

    bin_means = np.arange(step / 2, 1, step)

    in_bin_precisions = np.zeros(len(bin_means))

    for i, (bin_lower, bin_upper) in enumerate([(mean - step / 2, mean + step / 2) for mean in bin_means]):

        bin_mask = (top_probas >= bin_lower) & (top_probas < bin_upper)
        if np.any(bin_mask):
            in_bin_precisions[i] = np.mean(labels[bin_mask] == predictions[bin_mask])

    return bin_means, in_bin_precisions


def fit_temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find temperature scaling parameter through optimisting the expected calibration error.

    :param scores: Probabilities or logits of dimension samples x classes.
    :param labels: True labels as corresponding column indices
    :return: temperature scaling parameter lbda
    """

    probas = probas_from_logits(scores)

    res = minimize(ece_loss, 1, method='SLSQP', tol=1e-6, args=(probas, labels),
                   options={'maxiter': 10000}, bounds=((1e-10, 100),))

    assert res['success'] == True

    return res['x'][0]


