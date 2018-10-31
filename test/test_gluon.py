# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

from typing import List, Dict, Any
import pandas as pd
import mxnet as mx
from mxnet import nd, autograd
from mxnet import gluon
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import scipy.sparse

"""
https://mxnet.incubator.apache.org/api/python/gluon/nn.html

TODO:
- categorical data (sklearn.preprocessing.OrdinalEncoder) -> Embedding layer
- string data n-grams (hashed or tfidf) (sklearn.feature_extraction.text.TfidfVectorizer) -> Dense layer
- numerical data -> dense
"""

from pandas.api.types import is_float_dtype, is_string_dtype, is_integer_dtype


def encoder_for(df: pd.DataFrame, input_columns: List[str]) -> List[str]:
    blocks = []
    for col in input_columns:
        if is_integer_dtype(df[col]):
            blocks.append('cat')  # embedding
        elif is_float_dtype(df[col]):
            blocks.append('standardscaler')  # dense
        elif is_string_dtype(df[col]):
            blocks.append('tfidf')  # hashed, dense
    return blocks


class ConcatLayer(gluon.nn.HybridBlock):
    def __init__(self, inputs, n_outputs, **kwargs):
        """

        :param inputs: [{'name': 'a', 'layer': gluon.nn.Dense(3)}]
        :param n_outputs: int, number of labels
        :param kwargs:
        """
        gluon.nn.HybridBlock.__init__(self, **kwargs)

        self.inputs = inputs

        with self.name_scope():
            self.fc = gluon.nn.Dense(n_outputs)

            for inp in self.inputs:
                setattr(self, inp['name'], inp['layer'])

    def hybrid_forward(self, F, *args):
        assert len(args) == len(self.inputs)
        input_acts = [getattr(self, inp_config['name'])(arg) for inp_config, arg in zip(self.inputs, args)]
        input_acts_concat = F.concat(*input_acts, dim=1)
        fced = self.fc(input_acts_concat)
        return fced


def str_encode(data):
    return HashingVectorizer(dtype=np.float32, n_features=100000).fit_transform(data)


class Imputer(object):
    def __init__(self, input_columns: List[str], output_column: str):
        self.inputs = input_columns
        self.output = output_column

        # TODO
        self.data_ctx = mx.cpu()
        self.model_ctx = mx.cpu()
        # self.model_ctx = mx.gpu()

        self.model = None
        self.encoders = {
            # TODO
            # 'data': [],
            'label': None
        }

    # https://github.com/anttttti/Wordbatch/blob/master/wordbatch/batcher.py
    def __encode_data(self, df: pd.DataFrame, parallel: bool):
        def encode(data):
            if data.dtype == np.int:
                # categorical encoding
                return LabelBinarizer(sparse_output=True).fit_transform(data)
            else:
                # text encoding
                if parallel:
                    import multiprocessing
                    n = multiprocessing.cpu_count() - 1
                    pool = multiprocessing.Pool(n, maxtasksperchild=1)
                    jobs = pool.map(str_encode, np.array_split(data, n*2))
                    pool.close()
                    pool.join()
                    return scipy.sparse.vstack(jobs)
                else:
                    return str_encode(data)
        X_ = [encode(df[col]) for col in self.inputs]
        # fixme: should not hstack here but keep separate encodings and feed them to iterator
        X = scipy.sparse.hstack(X_).toarray()
        return X

    def fit(self, df: pd.DataFrame, batch_size: int, num_epochs: int, parallel_encodings: bool):
        """
        https://gluon.mxnet.io/chapter02_supervised-learning/softmax-regression-gluon.html
        """
        import time
        s = time.time()
        X = self.__encode_data(df, parallel_encodings)
        encode_duration = time.time() - s
        print('encoding took', encode_duration, 'seconds')

        lb = LabelBinarizer(sparse_output=False).fit(df[self.output])
        self.encoders['label'] = lb

        y = lb.transform(df[self.output]).astype(np.int32)

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        train_data = mx.io.NDArrayIter(
            mx.nd.array(X_train),
            mx.nd.array(y_train),
            batch_size=batch_size,
            last_batch_handle='discard'
        )

        test_data = mx.io.NDArrayIter(
            mx.nd.array(X_test),
            mx.nd.array(y_test),
            batch_size=batch_size,
            last_batch_handle='discard'
        )

        num_outputs = df[self.output].nunique()

        n_units_first = 3
        n_units_second = 5
        gluon_datawig = ConcatLayer(
            [
                {'name': 'a', 'layer': gluon.nn.Dense(n_units_first)},
                {'name': 'b', 'layer': gluon.nn.Dense(n_units_second)}
            ],
            num_outputs
        )
        gluon_datawig.hybridize(static_alloc=True,static_shape=True)
        gluon_datawig.collect_params().initialize(mx.init.Normal(sigma=1.), ctx=self.model_ctx)
        softmax_cross_entropy2 = gluon.loss.SoftmaxCrossEntropyLoss()
        trainer = gluon.Trainer(gluon_datawig.collect_params(), 'adam')  # , {'learning_rate': 0.1})
        print('net:', gluon_datawig)
        self.model = gluon_datawig

        # working, if input shape inference running against this code
        # batch = 32
        # input1 = nd.random.uniform(shape=[batch, 10])
        # input2 = nd.random.uniform(shape=[batch, 20])
        # output = gluon_datawig(input1, input2)
        # print(output.shape)

        print("-"*100)
        for e in range(num_epochs):
            cumulative_loss = 0
            train_data.reset()
            for i, batch in enumerate(train_data):
                data = batch.data[0].as_in_context(self.model_ctx)
                label = batch.label[0].as_in_context(self.model_ctx)
                label = label.argmax(axis=1)
                with autograd.record():
                    # fixme: should use separate encoded inputs here
                    output = gluon_datawig(data, data)
                    loss = softmax_cross_entropy2(output, label)

                loss.backward()
                trainer.step(batch_size)
                cumulative_loss += nd.sum(loss).asscalar()

            test_data.reset()
            train_data.reset()
            test_accuracy = self.__evaluate_accuracy(test_data, gluon_datawig)
            train_accuracy = self.__evaluate_accuracy(train_data, gluon_datawig)
            print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (
                e, cumulative_loss / df.shape[0], train_accuracy, test_accuracy))

    def predict_proba(self, df):
        X = self.__encode_data(df, parallel=False)
        data = mx.nd.array(X).as_in_context(self.model_ctx)
        # fixme: also here use separately encoded inputs instead
        predictions = self.model(data, data)
        s = predictions.softmax()
        # top_k = s.topk(k=3)
        decoded = [self.encoders['label'].classes_[int(i)] for i in s.argmax(axis=1).asnumpy().tolist()]
        print((decoded == df[self.output]).mean())
        return decoded, s.argmax(axis=1)

    def __evaluate_accuracy(self, data_iterator, net):
        acc = mx.metric.Accuracy()
        for i, batch in enumerate(data_iterator):
            data = batch.data[0].as_in_context(self.model_ctx)
            label = batch.label[0].as_in_context(self.model_ctx)
            label = label.argmax(axis=1)
            # fixme: also here, make use of separately encoded data
            output = net(data, data)
            predictions = nd.argmax(output, axis=1)
            acc.update(preds=predictions, labels=label)
        return acc.get()[1]


def test_full(data_frame):
    num_examples = 500
    n_labels = 3
    feature, label = 'feature', 'label'
    df = data_frame(n_samples=num_examples, feature_col=feature, label_col=label, num_labels=n_labels)
    df.loc[:, 'cat_feature'] = np.random.randint(1, 10, df.shape[0])
    # df.loc[df.label == df.label.unique()[0], 'cat_feature'] = 100
    # df.loc[df.label == df.label.unique()[1], 'cat_feature'] = 200
    print('finished data generation')
    i = Imputer(['feature', 'cat_feature'], 'label')
    # learning from cat feature only
    # i = Imputer(['cat_feature'], 'label')
    i.fit(df, batch_size=32, num_epochs=30, parallel_encodings=False)
    preds, probas = i.predict_proba(df)
    print(preds)
    print(probas)

