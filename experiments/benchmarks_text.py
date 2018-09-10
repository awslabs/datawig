import os
import shutil
import json
import numpy as np
import pandas as pd
from datawig import SimpleImputer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder


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
dir_path = os.path.dirname(os.path.realpath(__file__))


def impute_mean(X):
    return SimpleFill("mean").complete(X)


def impute_knn(X):
    # Use 3 nearest rows which have a feature to fill in each row's missing features
    return KNN(k=3).complete(X)


def impute_mice(X):
    return MICE().complete(X)


def impute_mf(X):
    return MatrixFactorization().complete(X)

def get_data(data):
    if data == 'finish_type_subset':
        df = pd.read_csv('./finish_val_data_all.csv')
    elif data == 'finish_type_full':
        df = pd.read_csv('./finish_attribute_only.csv')
    return df

def hash_text_encode_label(df, bucket_size=2**12):
    #uses same hashing vectorizer as DataWig's Imputer
    vectorizer = HashingVectorizer(n_features=bucket_size, ngram_range=(1,5), analyzer='char')
    tmp_col = pd.Series(index=df.index, data='')

    input_columns = ['text', 'title']
    for col in input_columns:
        tmp_col += col + " " + df[col].fillna("") + " "
    out = vectorizer.transform(tmp_col).astype(np.float32)
    out = out.toarray()

    #convert categorical labels into numbers
    lb_make = LabelEncoder()
    label = lb_make.fit_transform(df["finish"])
    out = np.concatenate((out, np.expand_dims(label, axis=1)), axis=1)

    df_encoded = pd.DataFrame(label, columns=['finish_encoded'])
    df_encoded = pd.concat([df, df_encoded], axis=1)

    return out, df_encoded

def run_text_datawig(df,imputer_fn, mask):
    input_cols = ['text', 'title']
    output_col = 'finish_encoded'

    #convert to strings for the Imputer to treat the numbers as categorical variables
    df['finish_encoded'] = df['finish_encoded'].astype(str)
    df_orig = df.copy()

    df.values[mask, 3] = np.nan
    idx_missing = df[output_col].isnull()
    output_path = os.path.join(dir_path, output_col)

    imputer = SimpleImputer(input_columns=input_cols,
                        output_column=output_col,
                        output_path=output_path).fit_hpo(train_df=df.loc[~idx_missing, :],
                                                         num_epochs=100,
                                                         patience=10,
                                                         batch_size=24,
                                                         num_hash_bucket_candidates=[2**12],
                                                         tokens_candidates=['char'],
                                                         learning_rate_candidates=[1e-2, 1e-3, 1e-4],
                                                         final_fc_hidden_units=[[], [100], [1024]]
                                                         )
    df_imputed = imputer.predict(df.loc[idx_missing, :].copy())

    predictions = df_imputed['finish_encoded_imputed'].values.astype(float)
    original = df_orig.values[mask,3].astype(float)
    mse = ((predictions-original) ** 2).mean()
    return mse


def run_imputation(X, mask, imputation_fn):
    # X is a data matrix which we're going to randomly drop entries from
    X_incomplete = X.copy()
    # missing entries indicated with NaN **ONLY IN LABEL COLUMN**
    label_column = X.shape[1]-1
    X_incomplete[mask, label_column]=np.nan
    #X_incomplete[mask] = np.nan
    X_imputed = imputation_fn(X_incomplete)
    mse = ((X_imputed[mask, label_column] - X[mask, label_column]) ** 2).mean()
    return mse


def experiment(percent_missing=10):
    MAE_DATA = [
        'finish_type_full'
    ]

    imputers = [
        impute_mean,
        impute_mf,
        impute_mice,
        impute_datawig
        impute_knn,
    ]

    results = []

    for data in MAE_DATA:
        print("*** Loading Data ***")
        df = get_data(data)
        missing_mask = np.random.rand(len(df)) < percent_missing / 100.
        print("*** Encoding Data ***")
        X, df_encoded = hash_text_encode_label(df)

        for imputer_fn in imputers:
            print("*** Running {} ***".format(imputer_fn.__name__))
            if imputer_fn == impute_datawig:
                mse = run_text_datawig(df_encoded, imputer_fn, missing_mask)
            else:
                mse = run_imputation(X, missing_mask, imputer_fn)

            result = {
                'data': data,
                'imputer': imputer_fn.__name__,
                'percent_missing': percent_missing,
                'mse': mse
            }
            print(result)
            results.append(result)

    return results

if __name__ == "__main__":
    results = experiment()
    json.dump(results, open(os.path.join(dir_path, 'benchmark_results.json'), 'w'))
