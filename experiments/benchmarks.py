import os
import sys
import shutil
import json
import itertools
import numpy as np
import pandas as pd
from datawig import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

sys.path.insert(0,'/Users/felix/code/datawig_fork')

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

def dict_product(hp_dict):
    '''
    Returns cartesian product of hyperparameters
    '''
    return [dict(zip(hp_dict.keys(),vals)) for vals in \
            itertools.product(*hp_dict.values())]

def evaluate_mse(X_imputed, X, mask):
    return ((X_imputed[mask] - X[mask]) ** 2).mean()

def fancyimpute_hpo(fancyimputer, param_candidates, X, mask, percent_validation=10):
    # first generate all parameter candidates for grid search
    all_param_candidates = dict_product(param_candidates)
    # get linear indices of all training data points
    train_idx = (mask.reshape(np.prod(X.shape)) == False).nonzero()[0]
    # get the validation mask
    n_validation = int(len(train_idx) * percent_validation/100)
    validation_idx = np.random.choice(train_idx,n_validation)
    validation_mask = np.zeros(np.prod(X.shape))
    validation_mask[validation_idx] = 1
    validation_mask = validation_mask.reshape(X.shape) > 0
    # save the original data
    X_incomplete = X.copy()
    # set validation and test data to nan
    X_incomplete[mask | validation_mask] = np.nan
    mse_hpo = []
    for params in all_param_candidates:
        X_imputed = fancyimputer(**params).fit_transform(X_incomplete)
        mse = evaluate_mse(X_imputed, X, validation_mask)
        print(f"Trained {fancyimputer.__name__} with {params}, mse={mse}")
        mse_hpo.append(mse)

    best_params = all_param_candidates[np.array(mse_hpo).argmin()]
    # now retrain with best params on all training data
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    X_imputed = fancyimputer(**best_params).fit_transform(X_incomplete)
    mse_best = evaluate_mse(X_imputed, X, mask)
    print(f"HPO: {fancyimputer.__name__}, best {best_params}, mse={mse_best}")
    return mse_best

def impute_mean(X, mask):
    return fancyimpute_hpo(SimpleFill,{'fill_method':["mean"]}, X, mask)

def impute_knn(X, mask, hyperparams={'k':[2,4,6]}):
    return fancyimpute_hpo(KNN,hyperparams, X, mask)

def impute_mf(X, mask, hyperparams={'rank':[5,10,50],'l2_penalty':[1e-3, 1e-5]}):
    return fancyimpute_hpo(MatrixFactorization, hyperparams, X, mask)

def impute_sklearn_rf(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = RandomForestRegressor(random_state=0)
    parameters = {
        'n_estimators': [2, 10, 100],
        'max_features;': [int(np.sqrt(X.shape[-1])), X.shape[-1]]
                }
    clf = GridSearchCV(reg, parameters, cv=5)
    X_pred = IterativeImputer(random_state=0, predictor=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse

def impute_sklearn_linreg(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    reg = LinearRegression()
    X_pred = IterativeImputer(random_state=0, predictor=reg).fit_transform(X_incomplete)
    mse = evaluate_mse(X_pred, X, mask)
    return mse

def impute_datawig(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    df = pd.DataFrame(X_incomplete)
    df.columns = [str(c) for c in df.columns]
    df = SimpleImputer.complete(df, hpo=True, verbose=0)
    mse = evaluate_mse(df.values, X, mask)
    return mse


def impute_datawig_iterative(X, mask):
    X_incomplete = X.copy()
    X_incomplete[mask] = np.nan
    df = pd.DataFrame(X_incomplete)
    df.columns = [str(c) for c in df.columns]
    df = SimpleImputer.complete(df, hpo=True, verbose=0, iterations=5)
    mse = evaluate_mse(df.values, X, mask)
    return mse

def get_data(data_fn):
    if data_fn.__name__ is 'make_low_rank_matrix':
        X = data_fn(n_samples=1000, n_features=10, effective_rank = 5, random_state=0)
    elif data_fn.__name__ is 'make_swiss_roll':
        X, t = data_fn(n_samples=1000, random_state=0)
        X = np.vstack([X.T, t]).T
    elif data_fn.__name__ in ['load_digits', 'load_wine', 'load_diabetes']:
        X, _ = data_fn(return_X_y=True)
    return X

def generate_missing_mask(X, percent_missing=10, missing_at_random=True):
    if missing_at_random:
        mask = np.random.rand(*X.shape) < percent_missing / 100.
    else:
        mask = np.zeros(X.shape)
        n_values_to_discard = int((percent_missing / 100) * X.shape[0])
        # for each affected column
        for col_affected in range(X.shape[1]):
            # select a random other column for missingness to depend on
            depends_on_col = np.random.choice([c for c in range(X.shape[1]) if c != col_affected])
            # pick a random percentile
            if n_values_to_discard < X.shape[0]:
                discard_lower_start = np.random.randint(0, X.shape[0]-n_values_to_discard)
            else:
                discard_lower_start = 0
            discard_idx = range(discard_lower_start, discard_lower_start + n_values_to_discard)
            values_to_discard = X[:,depends_on_col].argsort()[discard_idx]
            mask[values_to_discard, col_affected] = 1
    return mask > 0


def experiment(percent_missing_list=[10], nreps=1):
    DATA_LOADERS = [
        make_low_rank_matrix,
        load_diabetes,
        load_wine,
        make_swiss_roll
    ]

    imputers = [
        # impute_mean,
        # impute_knn,
        # impute_mf,
        # impute_sklearn_rf,
        # impute_sklearn_linreg,
        # impute_datawig
        impute_datawig_iterative
    ]

    results = []

    for percent_missing in percent_missing_list:
        for data_fn in DATA_LOADERS:
            X = get_data(data_fn)
            for missingness_at_random in [True, False]:
                for _ in range(nreps):
                    missing_mask = generate_missing_mask(X, percent_missing, missingness_at_random)
                    for imputer_fn in imputers:
                        mse = imputer_fn(X, missing_mask)
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

def run():

    # this appears to be neccessary for not running into too many open files errors
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))

    results = experiment(percent_missing_list=[5, 10, 30, 50, 70], nreps = 5)

    json.dump(results, open(os.path.join(dir_path, 'benchmark_results.json'), 'w'))

def plot_results(results):
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(results)
    df['mse_percent'] = df.mse / df.groupby(['data','missing_at_random','percent_missing'])['mse'].transform(max)
    df.groupby(['missing_at_random','percent_missing','imputer']).agg({'mse_percent':'median'})

    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    sns.boxplot(hue='imputer',y='mse_percent',x='percent_missing', data=df[df['missing_at_random']==True], notch=True)
    plt.title("Missing at random")
    plt.xlabel("% Samples Missing")
    plt.ylabel("Relative MSE in %")
    plt.legend()
    plt.subplot(2,1,2)
    sns.boxplot(hue='imputer',y='mse_percent',x='percent_missing', data=df[df['missing_at_random']==False], notch=True)
    plt.title("Missing not at random")
    plt.xlabel("% Samples Missing")
    plt.legend("")
    plt.ylabel("Relative MSE in %")

    plt.tight_layout()
    plt.savefig('benchmarks_datawig.pdf')

if __name__=='main':
    run()
