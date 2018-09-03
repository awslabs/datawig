import numpy as np
from scipy.optimize import minimize


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def compute_ece(scores, labels, temperature=1, step=.05):
    """

    """

    # compute probabilities applying tempered softmax
    probas = calibrate(scores, temperature)

    # probabilities for the most likely labels
    top_probas = np.max(probas, axis=1)

    # top prediction is independent of temperature scaling or logits/probas
    predictions = np.argmax(probas, 1)

    bin_means = np.arange(step / 2, 1, step)

    ece = 0

    for bin_lower, bin_upper in [(mean - step / 2, mean + step / 2) for mean in bin_means]:

        bin_mask = (top_probas >= bin_lower) & (top_probas < bin_upper)
        if np.any(bin_mask):
            in_bin_confidence = np.mean(top_probas[bin_mask])
            in_bin_precision = np.mean(labels[bin_mask] == predictions[bin_mask])
            ece += np.abs(in_bin_confidence - in_bin_precision) * np.mean(bin_mask)

    return ece


def ece_loss(temperature, *args):
    """
    wrapper around ece to be call during optimisation

    """

    scores, labels = args

    return compute_ece(scores, labels, temperature=temperature)


def logits_from_probas(probas, force=False):
    """
    Returns logits for a vector of class probabilities. This is not a unique transformation.
    If the input is not a probability, no transformation is made.
    """

    # check whether rows of input are probabilities
    if (force is True) or (np.all(np.sum(probas, 1) - 1 < 1e-12) and np.all(probas <= 1)):
        return np.log(probas)
    else:
        return probas


def probas_from_logits(scores, temperature=1, force=False):
    """
    Returns probabilitiess for a vector of class logits.
    If the input is a probability, no transformation is made.
    """

    # check whether rows of input are probabilities
    if np.all(np.sum(scores, 1) - 1 < 1e-2) and np.all(scores <= 1) and force is False:
        return scores
    else:
        return np.array([softmax(temperature * row) for row in scores])


def calibrate(scores, temperature):

    logits = logits_from_probas(scores, force=True)

    return np.array([softmax(temperature * row) for row in logits])


def reliability(scores, labels, step=.05):

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


def fit_temperature(scores, labels):

    probas = probas_from_logits(scores)

    res = minimize(ece_loss, 1, method='SLSQP', tol=1e-6, args=(probas, labels),
                   options={'maxiter': 10000}, bounds=((1e-10, 100),))

    assert res['success'] == True

    return res['x'][0]


