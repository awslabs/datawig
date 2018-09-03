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


def test_automatic_calibration():
    """
    Fit a simple model with synthetic data and assert that calibration improves the expected calibration error
    """

    feature_col = "string_feature"
    categorical_col = "categorical_feature"
    label_col = "label"

    n_samples = 2000
    num_labels = 3
    seq_len = 20
    vocab_size = int(2 ** 10)

    latent_dim = 30
    embed_dim = 30

    # generate some random data
    random_data = generate_string_data_frame(feature_col=feature_col,
                                             label_col=label_col,
                                             vocab_size=vocab_size,
                                             num_labels=num_labels,
                                             num_words=seq_len,
                                             n_samples=n_samples)

    # we use a the label prefixes as a dummy categorical input variable
    random_data[categorical_col] = random_data[label_col].apply(lambda x: x[:2])

    df_train, df_test, df_val = random_split(random_data, [.8, .1, .1])

    data_encoder_cols = [
        BowEncoder(feature_col, feature_col + "_bow", max_tokens=vocab_size),
        SequentialEncoder(feature_col, feature_col + "_lstm", max_tokens=vocab_size, seq_len=seq_len),
        CategoricalEncoder(categorical_col, max_tokens=num_labels)
    ]
    label_encoder_cols = [CategoricalEncoder(label_col, max_tokens=num_labels)]

    data_cols = [
        BowFeaturizer(
            feature_col + "_bow",
            vocab_size=vocab_size),
        LSTMFeaturizer(
            field_name=feature_col + "_lstm",
            seq_len=seq_len,
            latent_dim=latent_dim,
            num_hidden=30,
            embed_dim=embed_dim,
            num_layers=2,
            vocab_size=num_labels),
        EmbeddingFeaturizer(
            field_name=categorical_col,
            embed_dim=embed_dim,
            vocab_size=num_labels)
    ]

    output_path = os.path.join(dir_path, "resources", "tmp", "imputer_experiment_synthetic_data")

    num_epochs = 20
    batch_size = 32
    learning_rate = 1e-2

    imputer = Imputer(
        data_featurizers=data_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path=output_path
    ).fit(
        train_df=df_train,
        test_df=df_val,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size
    )

    assert imputer.calibration_info['ece_pre'] > imputer.calibration_info['ece_post']


