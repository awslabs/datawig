import datawig
from datawig.simple_imputer import SimpleImputer
from datawig.utils import logger
from pandas import read_csv
from datawig.utils import random_split
from datawig.utils import logger
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import seaborn as sns
from scipy.stats import entropy
logger.setLevel("DEBUG")

def test_coo():

    # Load coo data
    df = pd.read_csv('./../examples/data/coo_audit.csv')
    df = df.dropna(subset=['description', 'coo_value_manual', 'manufacturer.value'])


    # Specify encoders and featurizers
    data_encoder_cols = [datawig.column_encoders.BowEncoder('manufacturer.value', tokens="words"),
                         datawig.column_encoders.TfIdfEncoder('description', tokens="words"),
                         datawig.column_encoders.CategoricalEncoder('brand_name', max_tokens=1e3)]
    data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('manufacturer.value'),
                            datawig.mxnet_input_symbols.BowFeaturizer('description'),
                            datawig.mxnet_input_symbols.EmbeddingFeaturizer('brand_name')]

    label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('coo_value_manual')]


    # Specify model
    imputer = datawig.Imputer(
       data_featurizers=data_featurizer_cols,
       label_encoders=label_encoder_cols,
       data_encoders=data_encoder_cols,
       output_path='/tmp'
       )


    # Train
    tr, te = random_split(df.sample(100), [.8, .2])
    imputer.fit(train_df=tr, test_df=te, num_epochs=20)
    predictions = imputer.predict(te)


    # Evaluate
    prec = precision_score(te.coo_value_manual, te.coo_value_manual_imputed, average='weighted')

    print(imputer.explain('DE', 20, 'coo_value_manual'))

    return imputer


def test_simulated():

    # Generate simulated data for testing explain method
    # Predict output column with entries in ['foo', 'bar'] from two columns, one
    # categorical in ['foo', 'dummy'], one text in ['text_foo_text', 'text_dummy_text'].
    # the output column is deterministically 'foo', if 'foo' occurs anywhere in any input column.
    N = 100
    cat_in_col = ['foo' if r > (1 / 3) else 'dummy' for r in np.random.rand(N)]
    text_in_col = ['textFtext' if r > (1 / 3) else 'textDtext' for r in np.random.rand(N)]
    cat_out_col = ['foo' if 'foo' in input[0] + input[1] else 'bar' for input in zip(cat_in_col, text_in_col)]

    df = pd.DataFrame()
    df['in_cat'] = cat_in_col
    df['in_text'] = text_in_col
    df['out_cat'] = cat_out_col


    # Specify encoders and featurizers # Todo: add column with HasingVectorizer
    data_encoder_cols = [datawig.column_encoders.TfIdfEncoder('in_text', tokens="chars"),
                         datawig.column_encoders.CategoricalEncoder('in_cat', max_tokens=1e1)]
    data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('in_text'),
                            datawig.mxnet_input_symbols.EmbeddingFeaturizer('in_cat')]

    label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('out_cat')]


    # Specify model
    imputer = datawig.Imputer(
       data_featurizers=data_featurizer_cols,
       label_encoders=label_encoder_cols,
       data_encoders=data_encoder_cols,
       output_path='/tmp'
       )

    # Train
    tr, te = random_split(df.sample(50), [.8, .2])
    imputer.fit(train_df=tr, test_df=te, num_epochs=5)
    predictions = imputer.predict(te)

    # Evaluate
    prec = precision_score(te.out_cat, te.out_cat_imputed, average='weighted')

    return imputer

# imputer = test_coo()
imputer = test_simulated()
imputer.explain('foo', 5, 'out_cat')

temp = 1