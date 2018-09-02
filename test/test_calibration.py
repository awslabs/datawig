import numpy as np
from scipy.optimize import minimize
from datawig import calibration


def generate_synthetic_data(K=None, N=None, p_correct=None):
    """
    Generate synthetic data of probabili

    :param K: number of classes
    :type K: int
    :param N: number of samples
    :type N: int
    :param p_correct: fraction of correct predictions
    :type p_correct:
    :return:
    :rtype:
    """

    if p_correct is None:
        p_correct = .8

    if K is None:
        K = 5

    if N is None:
        N = 100

    # generate labels
    train_labels = np.array([np.random.randint(5) for n in range(N)])

    # generate features
    train_data = np.empty([N, K])

    for n in range(N):
        # pick correct label with probability p_correct
        if np.random.rand() < p_correct:
            label = train_labels[n]
        else:
            label = np.random.choice([val for val in range(K) if val != train_labels[n]])

        # assign logits from uniform [1,2] for correct and uniform [0,1] for incorrect labels
        for k in range(K):
            if label == k:
                train_data[n, k] = np.random.uniform(1, 2)
            else:
                train_data[n, k] = np.random.uniform(0, 1)

        train_data[n, :] = calibration.softmax(train_data[n, :])

    # assert probabilities sum to 1
    assert np.all((np.sum(train_data, 1) - 1) < 1e-10)

    return train_data, train_labels


def test_calibration():
    train_data, train_labels = generate_synthetic_data(p_correct=.7, N=50, K=10)

    temperature = calibration.fit_temperature(train_data, train_labels)

    assert calibration.compute_ece(train_data, train_labels, temperature=temperature) < \
           calibration.compute_ece(train_data, train_labels)
