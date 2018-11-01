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

"""

DataWig MxNet input symbols:
extract features or pass on encoded numerical representations of rows

"""

from typing import Any, List

import mxnet as mx

from .utils import get_context


class Featurizer(object):
    """
    Featurizer for data that is encoded into numerical format with ColumnEncoder
    Is used to feed data into mxnet compute graph

    :param field_name: Field name of featurizer output for mxnet variable/symbols
    :param latent_dim: Dimensionality of resulting features
    """

    def __init__(self,
                 field_name: str,
                 latent_dim: int) -> None:
        self.field_name = field_name
        self.latent_dim = int(latent_dim)
        self.input_symbol = mx.sym.Variable("{}".format(field_name))
        self.prefix = self.field_name + "_"
        self.symbol = None

    def latent_symbol(self) -> mx.symbol:
        """

        Returns mxnet Featurizer symbol
        :return: Featurizer as mx.symbol

        """
        return self.symbol


class NumericalFeaturizer(Featurizer):
    """

    NumericFeaturizer, a one hidden layer neural network with relu activations

    :param field_name: name of the column
    :param numeric_latent_dim: number of hidden units
    :param numeric_hidden_layers: number of hidden layers
    :return:
    """

    def __init__(self,
                 field_name: str,
                 numeric_latent_dim: int = 100,
                 numeric_hidden_layers: int = 1) -> None:
        super(NumericalFeaturizer, self).__init__(field_name, numeric_latent_dim)

        self.numeric_hidden_layers = int(numeric_hidden_layers)
        self.numeric_latent_dim = int(numeric_latent_dim)

        with mx.name.Prefix(self.prefix):
            self.symbol = self.input_symbol
            for _ in range(self.numeric_hidden_layers):
                symbol = mx.sym.FullyConnected(
                    data=self.symbol,
                    num_hidden=self.numeric_latent_dim
                )
                self.symbol = mx.symbol.Activation(data=symbol, act_type="relu")


class LSTMFeaturizer(Featurizer):
    """

    LSTMFeaturizer maps an input representing a sequence of symbols in [0, vocab_size] of shape
    (batch, seq_len) into a latent vector of shape (batch, latent_dim).
    The featurization is done with num_layers LSTM that has num_layers layers and num_hidden units

    :param field_name: input symbol
    :param seq_len: length of sequence
    :param vocab_size: size of vocabulary
    :param num_hidden: number of hidden units
    :param num_layers: number of layers
    :param latent_dim: latent dimensionality (number of hidden units in fully connected
                        output layer of lstm)

    """

    def __init__(self,
                 field_name: str,
                 seq_len: int = 500,
                 max_tokens: int = 40,
                 embed_dim: int = 50,
                 num_hidden: int = 50,
                 num_layers: int = 2,
                 latent_dim: int = 50,
                 use_gpu: bool = False if mx.cpu() in get_context() else True) -> None:
        super(LSTMFeaturizer, self).__init__(field_name, latent_dim)

        self.vocab_size = int(max_tokens)
        self.embed_dim = int(embed_dim)
        self.seq_len = int(seq_len)
        self.num_hidden = int(num_hidden)
        self.num_layers = int(num_layers)

        with mx.name.Prefix(field_name + "_"):
            embed_symbol = mx.sym.Embedding(
                data=self.input_symbol,
                input_dim=self.vocab_size,
                output_dim=self.embed_dim
            )

            def make_cell(layer_index):
                prefix = 'lstm_l{}_{}'.format(layer_index, self.field_name)
                cell_type = mx.rnn.FusedRNNCell if use_gpu else mx.rnn.LSTMCell
                cell = cell_type(num_hidden=self.num_hidden, prefix=prefix)
                # residual connection can only be applied in the first layer currently in mxnet
                return cell if layer_index == 0 else mx.rnn.ResidualCell(cell)

            stack = mx.rnn.SequentialRNNCell()
            for i in range(self.num_layers):
                stack.add(make_cell(layer_index=i))
            output, _ = stack.unroll(self.seq_len, inputs=embed_symbol, merge_outputs=True)

            self.symbol = mx.sym.FullyConnected(data=output, num_hidden=self.latent_dim)


class EmbeddingFeaturizer(Featurizer):
    """

    EmbeddingFeaturizer for categorical data

    :param field_name: name of the column
    :param vocab_size: size of the vocabulary, defaults to 100
    :param embed_dim: dimensionality of embedding, defaults to 10

    """

    def __init__(self,
                 field_name: str,
                 max_tokens: int = 100,
                 embed_dim: int = 10) -> None:
        super(EmbeddingFeaturizer, self).__init__(field_name, embed_dim)

        self.vocab_size = int(max_tokens)
        self.embed_dim = int(embed_dim)

        with mx.name.Prefix(field_name + "_"):
            symbol = mx.sym.Embedding(
                data=self.input_symbol,
                input_dim=self.vocab_size,
                output_dim=self.embed_dim
            )
            self.symbol = mx.sym.FullyConnected(data=symbol, num_hidden=self.latent_dim)


class BowFeaturizer(Featurizer):
    """

    Bag of words Featurizer for string data

    :param field_name: name of the column
    :param vocab_size: size of the vocabulary (number of hash buckets), defaults to 2**15

    """

    def __init__(self,
                 field_name: str,
                 max_tokens: int = 2 ** 15) -> None:
        super(BowFeaturizer, self).__init__(field_name, max_tokens)

        with mx.name.Prefix(field_name + "_"):
            self.symbol = mx.sym.Variable("{}".format(field_name), stype='csr')
