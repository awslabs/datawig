import numpy as np
import pandas as pd
from datawig import SimpleImputer

from sklearn.datasets import (
    make_low_rank_matrix,
    load_diabetes,
    load_wine,
    make_swiss_roll
    )

from fancyimpute import (
    MatrixFactorization,
    MICE,
    KNN,
    SimpleFill
)

np.random.seed(0)


def impute_mean(X):
    return SimpleFill("mean").complete(X)

def impute_knn(X):
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    return KNN(k=3).complete(X)

def impute_mice(X):
    return MICE().complete(X)

def impute_mf(X):
    return MatrixFactorization().complete(X)

def impute_datawig(X):
    df = pd.DataFrame(X)
    df.columns = [str(c) for c in df.columns]
    df_imputed = {}
    for output_col in df.columns:
        input_cols = sorted(list(set(df.columns) - set([output_col])))
        idx_missing = df[output_col].isnull()
        imputer = SimpleImputer(input_columns = input_cols,
                                output_column = output_col) \
                                .fit(df.loc[~idx_missing, :])
                                # .fit_hpo(df.loc[~idx_missing,:])
        df_imputed[output_col] = imputer.predict(df.loc[idx_missing,:])

    for output_col in df.columns:
        idx_missing = df[output_col].isnull()
        df.loc[idx_missing, output_col] = \
            df_imputed[output_col].loc[idx_missing, output_col + "_imputed"]

    return df.values


def get_data(data_fn):
    if data_fn.__name__ is 'make_low_rank_matrix':
        X = data_fn(n_samples=1000, random_state=0)
    elif data_fn.__name__ is 'make_swiss_roll':
        X, t = data_fn(n_samples=1000, random_state=0)
        X = np.vstack([X.T, t]).T
    elif data_fn.__name__ in ['load_digits', 'load_wine', 'load_diabetes']:
        X, _ = data_fn(return_X_y=True)
    return X

def run_imputation(X, mask, imputation_fn):

    # X is a data matrix which we're going to randomly drop entries from
    X_incomplete = X.copy()
    # missing entries indicated with NaN
    X_incomplete[mask] = np.nan
    X_imputed = imputation_fn(X_incomplete)
    mse = ((X_imputed[mask] - X[mask]) ** 2).mean()
    return mse


def experiment(percent_missing=10):

    DATA_LOADERS = [
        make_low_rank_matrix,
        load_diabetes,
        load_wine,
        make_swiss_roll
        ]

    imputers = [
        impute_mean,
        impute_knn,
        impute_mice,
        impute_mf,
        impute_datawig
    ]

    results = []

    for data_fn in DATA_LOADERS:
        X = get_data(data_fn)
        missing_mask = np.random.rand(*X.shape) < percent_missing / 100.
        for imputer_fn in imputers:
            mse = run_imputation(X, missing_mask, imputer_fn)
            result = {
                'data': data_fn.__name__,
                'imputer': imputer_fn.__name__,
                'percent_missing': percent_missing,
                'mse': mse
            }
            print(result)
            results.append(result)
    return results
