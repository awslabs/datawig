import datawig
from datawig.simple_imputer import SimpleImputer
from datawig.utils import logger
from pandas import read_csv
from datawig.utils import random_split
from datawig.utils import logger
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
# import seaborn as sns
from scipy.stats import entropy
logger.setLevel("INFO")
#
# def test_coo():
#
#     # Load coo data
#     df = pd.read_csv('./../examples/data/coo_audit.csv')
#     df = df.dropna(subset=['description', 'coo_value_manual', 'manufacturer.value'])
#
#
#     # Specify encoders and featurizers
#     data_encoder_cols = [datawig.column_encoders.BowEncoder('manufacturer.value', tokens="words"),
#                          datawig.column_encoders.TfIdfEncoder('description', tokens="words"),
#                          datawig.column_encoders.CategoricalEncoder('brand_name', max_tokens=1e3)]
#     data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('manufacturer.value'),
#                             datawig.mxnet_input_symbols.BowFeaturizer('description'),
#                             datawig.mxnet_input_symbols.EmbeddingFeaturizer('brand_name')]
#
#     label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('coo_value_manual')]
#
#
#     # Specify model
#     imputer = datawig.Imputer(
#        data_featurizers=data_featurizer_cols,
#        label_encoders=label_encoder_cols,
#        data_encoders=data_encoder_cols,
#        output_path='/tmp'
#        )
#
#
#     # Train
#     tr, te = random_split(df.sample(50), [.8, .2])
#     imputer.fit(train_df=tr, test_df=te, num_epochs=5)
#     predictions = imputer.predict(te)
#
#
#     # Evaluate
#     prec = precision_score(te.coo_value_manual, te.coo_value_manual_imputed, average='weighted')
#
#     print(imputer.explain('DE', 20, 'coo_value_manual'))
#
#
# def test_simulated():
#
#     # Generate simulated data for testing explain method
#     # Predict output column with entries in ['foo', 'bar'] from two columns, one
#     # categorical in ['foo', 'dummy'], one text in ['text_foo_text', 'text_dummy_text'].
#     # the output column is deterministically 'foo', if 'foo' occurs anywhere in any input column.
#     N = 100
#     cat_in_col = ['foo' if r > (1 / 2) else 'dummy' for r in np.random.rand(N)]
#     text_in_col = ['text_foo_text' if r > (1 / 2) else 'text_dummy_text' for r in np.random.rand(N)]
#     cat_out_col = ['foo' if 'foo' in input[0] + input[1] else 'bar' for input in zip(cat_in_col, text_in_col)]
#
#     df = pd.DataFrame()
#     df['in_cat'] = cat_in_col
#     df['in_text'] = text_in_col
#     df['out_cat'] = cat_out_col
#
#
#     # Specify encoders and featurizers # Todo: add column with HasingVectorizer
#     data_encoder_cols = [datawig.column_encoders.TfIdfEncoder('in_text', tokens="chars")]
#                          # datawig.column_encoders.CategoricalEncoder('in_cat', max_tokens=1e1)]
#     data_featurizer_cols = [datawig.mxnet_input_symbols.BowFeaturizer('in_text')]
#                             # datawig.mxnet_input_symbols.EmbeddingFeaturizer('in_cat')]
#
#     label_encoder_cols = [datawig.column_encoders.CategoricalEncoder('out_cat')]
#
#
#     # Specify model
#     imputer = datawig.Imputer(
#        data_featurizers=data_featurizer_cols,
#        label_encoders=label_encoder_cols,
#        data_encoders=data_encoder_cols,
#        output_path='/tmp'
#        )
#
#     # Train
#     tr, te = random_split(df.sample(50), [.8, .2])
#     imputer.fit(train_df=tr, test_df=te, num_epochs=5)
#     predictions = imputer.predict(te)
#
#     # Evaluate
#     prec = precision_score(te.out_cat, te.out_cat_imputed, average='weighted')
#
#     return imputer
#
# # imputer = test_simulated()
# # exp = imputer.explain('foo', 10, 'out_cat')
# #
# # a = 1


def test_module():
    import logging
    logging.getLogger().setLevel(logging.INFO)
    import mxnet as mx
    import numpy as np

    mx.random.seed(1234)
    fname = mx.test_utils.download(
        'https://s3.us-east-2.amazonaws.com/mxnet-public/letter_recognition/letter-recognition.data')
    data = np.genfromtxt(fname, delimiter=',')[:, 1:]
    label = np.array([ord(l.split(',')[0]) - ord('A') for l in open(fname, 'r')])

    batch_size = 32
    ntrain = int(data.shape[0] * 0.8)
    train_iter = mx.io.NDArrayIter(data[:ntrain, :], label[:ntrain], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(data[ntrain:, :], label[ntrain:], batch_size)


    import mxnet as mx
    net = mx.sym.Variable('data')
    net = mx.sym.FullyConnected(net, name='fc1', num_hidden=64)
    net = mx.sym.SoftmaxOutput(net, name='softmax')

    mod = mx.mod.Module(symbol=net,
                        context=mx.cpu(),
                        data_names=['data'],
                        label_names=['softmax_label'])
    # fit the module
    mod.fit(train_iter,
            eval_data=val_iter,
            optimizer='sgd',
            optimizer_params={'learning_rate':0.1},
            eval_metric='acc',
            num_epoch=8)
    val_iter.reset()
    a = mod.predict(val_iter)
    # val_iter.reset()
    # a = mod.score(val_iter)
    a = 1
