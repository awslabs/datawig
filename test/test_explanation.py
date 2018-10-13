import datawig
from datawig.utils import random_split
from datawig.utils import logger
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score
logger.setLevel("DEBUG")


def test_explain_method_synthetic():

    # Generate simulated data for testing explain method
    # Predict output column with entries in ['foo', 'bar'] from two columns, one
    # categorical in ['foo', 'dummy'], one text in ['text_foo_text', 'text_dummy_text'].
    # the output column is deterministically 'foo', if 'foo' occurs anywhere in any input column.
    N = 100
    cat_in_col = ['foo' if r > (1 / 2) else 'dummy' for r in np.random.rand(N)]
    text_in_col = ['fff' if r > (1 / 2) else 'ddd' for r in np.random.rand(N)]
    hash_in_col = ['h' for r in range(N)]
    cat_out_col = ['foo' if 'f' in input[0] + input[1] else 'bar' for input in zip(cat_in_col, text_in_col)]

    df = pd.DataFrame()
    df['in_cat'] = cat_in_col
    df['in_text'] = text_in_col
    df['in_text_hash'] = hash_in_col
    df['out_cat'] = cat_out_col

    # Specify encoders and featurizers #
    data_encoder_cols = [datawig.column_encoders.TfIdfEncoder('in_text', tokens="chars"),
                         datawig.column_encoders.CategoricalEncoder('in_cat', max_tokens=1e1),
                         datawig.column_encoders.BowEncoder('in_text_hash', tokens="chars")]
    data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('in_text'),
                            datawig.mxnet_input_symbols.EmbeddingFeaturizer('in_cat'),
                            datawig.mxnet_input_symbols.BowFeaturizer('in_text_hash')]

    label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('out_cat')]

    # Specify model
    imputer = datawig.Imputer(
       data_featurizers=data_featurizer_cols,
       label_encoders=label_encoder_cols,
       data_encoders=data_encoder_cols,
       output_path='/tmp'
       )

    # Train
    tr, te = random_split(df.sample(90), [.8, .2])
    imputer.fit(train_df=tr, test_df=te, num_epochs=30, learning_rate = 1e-2)
    imputer.predict(te)

    # Evaluate
    assert precision_score(te.out_cat, te.out_cat_imputed, average='weighted') > .99

    # assert explanations
    assert np.all(['f' in token for token, weight in imputer.explain('foo')['in_text']][:3])
    assert ['f' in token for token, weight in imputer.explain('foo')['in_cat']][0]

