import os
import shutil
import json
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
    IterativeImputer,
    BiScaler,
    KNN,
    SimpleFill
)

np.random.seed(0)
dir_path = '.'


def impute_mean(X):
    return SimpleFill("mean").fit_transform(X)


def impute_knn(X):
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    return KNN(k=3).fit_transform(X)


def impute_mf(X):
    return MatrixFactorization().fit_transform(X)


def impute_datawig(X):
    df = pd.DataFrame(X)
    df.columns = [str(c) for c in df.columns]
    # from datawig.utils import logger
    # logger.setLevel("ERROR")
    
    # df_imputed = {}
    # for output_col in df.columns:
    #     input_cols = sorted(list(set(df.columns) - set([output_col])))
    #     idx_missing = df[output_col].isnull()
    #     output_path = os.path.join(dir_path, output_col)
    #     imputer = SimpleImputer(input_columns=input_cols,
    #                             output_column=output_col,
    #                             output_path=output_path).fit(df.loc[~idx_missing, :], num_epochs=50, patience=5)
    #     df_imputed[output_col] = imputer.predict(df.loc[idx_missing, :])
    #     shutil.rmtree(output_path)

    # for output_col in df.columns:
    #     idx_missing = df[output_col].isnull()
    #     df.loc[idx_missing, output_col] = \
    #         df_imputed[output_col].loc[idx_missing, output_col + "_imputed"]
    df = SimpleImputer.complete(df)
#     ((X_imputed[mask] - X[mask]) ** 2).mean()
    return df.values


def get_data(data_fn):
    if data_fn.__name__ is 'make_low_rank_matrix':
        X = data_fn(n_samples=1000, n_features=10, effective_rank = 5, random_state=0)
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

def generate_missing_mask(X, percent_missing=10, missing_at_random=True):
    if missing_at_random:
        mask = np.random.rand(*X.shape) < percent_missing / 100.
    else:
        mask = np.zeros(X.shape)
        # select a random number of columns affected by the missingness
        n_cols_affected = np.random.randint(2,X.shape[1])
        # select a random set of columns
        cols_permuted = np.random.permutation(range(X.shape[1]))
        cols_affected = cols_permuted[:n_cols_affected]
        cols_unaffected = cols_permuted[n_cols_affected:]
        # for each affected column
        for col_affected in cols_affected:
            # select a random other column for missingness to depend on
            depends_on_col = np.random.choice(cols_unaffected)
            # pick a random percentile
            n_values_to_discard = X.shape[0] // n_cols_affected
            discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard-1)
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return mask > 0
       

def experiment(percent_missing_list=[10]):
    DATA_LOADERS = [
        make_low_rank_matrix,
        load_diabetes,
        load_wine,
        make_swiss_roll
    ]

    imputers = [
        impute_mean,
        impute_knn,
        impute_mf,
        impute_datawig
    ]

    results = []

    for percent_missing in percent_missing_list:
        for data_fn in DATA_LOADERS:
            X = get_data(data_fn)
            for missingness_at_random in [True, False]:
                missing_mask = generate_missing_mask(X, percent_missing, missingness_at_random)
                for imputer_fn in imputers:
                    mse = run_imputation(X, missing_mask, imputer_fn)
                    result = {
                        'data': data_fn.__name__,
                        'imputer': imputer_fn.__name__,
                        'percent_missing': percent_missing,
                        'missing_at_random': missingness_at_random,
                        'mse': mse
                    }
                    print(result)
                    results.append(result)
    return results


import resource
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
# results = experiment(percent_missing_list=[10, 30, 50])
# json.dump(results, open(os.path.join(dir_path, 'benchmark_results.json'), 'w'))
