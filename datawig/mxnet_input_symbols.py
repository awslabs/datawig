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

from typing import List, Any
import mxnet as mx

from .utils import gpu_device


class Featurizer():
    '''

    Featurizer for data that is encoded into numerical format with ColumnEncoder
    Is used to feed data into mxnet compute graph

    :param field_name: Field name of featurizer output for mxnet variable/symbols
    :param latent_dim: Dimensionality of resulting features

    '''

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


class ImageFeaturizer(Featurizer):
    '''

    ImageFeaturizer extracts image features given an image using a standard network architecture

    :param field_name: name of the column
    :param fine_tune: boolean that determines if entire image feature extraction model will be tuned
    :layer_dim: list containing dimensions of fully connected layer after fwd pass through image
                extraction network. The length of this list corresponds to the number of FC layers,
                and the contents of the list are integers with corresponding hidden layer size

    '''

    def __init__(self,
                 field_name: str,
                 fine_tune: bool = False,
                 layer_dim: List[int] = None,
                 model_name: str = 'densenet121') -> None:
        if not layer_dim:
            layer_dim = [1024]

        super(ImageFeaturizer, self).__init__(field_name, layer_dim[-1])

        self.fine_tune = fine_tune
        self.model_name = model_name
        self.layer_dim = [int(x) for x in layer_dim]

        with mx.name.Prefix(self.prefix):
            symbol, arg_params = self.__get_pretrained_model(self.input_symbol,
                                                             self.model_name)
            self.params = arg_params

            for hidden_dim in self.layer_dim:
                symbol = mx.sym.FullyConnected(
                    data=symbol,
                    num_hidden=hidden_dim
                )
                symbol = mx.symbol.Activation(data=symbol, act_type="softrelu")

            self.symbol = symbol

    @staticmethod
    def __get_pretrained_model(input_symbol: mx.symbol,
                               model_name: str = 'densenet121') -> Any:
        '''

        Loads a pretrained model from gluon model zoo

        :param input_symbol: mxnet symbol Variable with the associated name of the column
        :model_name: string containing the gluon model name for loading the pretrained model
                     currently supports 'densenet121', resnet50_v2, alexnet, and squeezenet1.0
                      and 'resnet18_v2' (default: densenet121)
        :return mxnet symbol, params dictionary

        '''

        image_network_pretrained = mx.gluon.model_zoo.vision.get_model(model_name,
                                                                       prefix='image_featurizer_',
                                                                       ctx=mx.cpu(),
                                                                       pretrained=True)

        internals = image_network_pretrained(input_symbol).get_internals()

        supported_models = ['densenet121',
                            'resnet18_v2',
                            'resnet50_v2',
                            'alexnet',
                            'squeezenet1.0']

        if model_name in supported_models:
            outputs = [internals['image_featurizer_flatten0_flatten0_output']]
        elif model_name == 'vgg16':
            outputs = [internals['image_featurizer_dense0_relu_fwd_output']]
        else:
            raise ValueError(
                "Only {} are supported for models, got {}".format(", ".join(supported_models),model_name))

        # feed one random input through network
        feat_model = mx.gluon.SymbolBlock(outputs, input_symbol,
                                          params=image_network_pretrained.collect_params())
        _ = feat_model(mx.nd.random.normal(shape=(16, 3, 224, 224)))

        # convert to numpy
        sym = feat_model(input_symbol)
        args = {}
        for key, value in feat_model.collect_params().items():
            args[key] = mx.nd.array(value.data().asnumpy())

        return sym, args


class NumericalFeaturizer(Featurizer):
    '''

    NumericFeaturizer, a one hidden layer neural network with relu activations

    :param field_name: name of the column
    :param latent_dim: number of hidden units
    :param hidden_layers: number of hidden layers
    :return:
    '''

    def __init__(self,
                 field_name: str,
                 latent_dim: int = 100,
                 hidden_layers: int = 1) -> None:
        super(NumericalFeaturizer, self).__init__(field_name, latent_dim)

        self.hidden_layers = int(hidden_layers)

        with mx.name.Prefix(self.prefix):
            self.symbol = self.input_symbol
            for _ in range(self.hidden_layers):
                symbol = mx.sym.FullyConnected(
                    data=self.symbol,
                    num_hidden=self.latent_dim
                )
                self.symbol = mx.symbol.Activation(data=symbol, act_type="relu")


class LSTMFeaturizer(Featurizer):
    '''

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

    '''

    def __init__(self,
                 field_name: str,
                 seq_len: int = 500,
                 vocab_size: int = 40,
                 embed_dim: int = 50,
                 num_hidden: int = 50,
                 num_layers: int = 2,
                 latent_dim: int = 50,
                 use_gpu: bool = mx.gpu() if gpu_device() else mx.cpu()) -> None:
        super(LSTMFeaturizer, self).__init__(field_name, latent_dim)

        self.vocab_size = int(vocab_size)
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
    '''

    EmbeddingFeaturizer for categorical data

    :param field_name: name of the column
    :param vocab_size: size of the vocabulary, defaults to 100
    :param embed_dim: dimensionality of embedding, defaults to 10

    '''

    def __init__(self,
                 field_name: str,
                 vocab_size: int = 100,
                 embed_dim: int = 10) -> None:
        super(EmbeddingFeaturizer, self).__init__(field_name, embed_dim)

        self.vocab_size = int(vocab_size)
        self.embed_dim = int(embed_dim)

        with mx.name.Prefix(field_name + "_"):
            symbol = mx.sym.Embedding(
                data=self.input_symbol,
                input_dim=self.vocab_size,
                output_dim=self.embed_dim
            )
            self.symbol = mx.sym.FullyConnected(data=symbol, num_hidden=self.latent_dim)


class BowFeaturizer(Featurizer):
    '''

    Bag of words Featurizer for string data

    :param field_name: name of the column
    :param vocab_size: size of the vocabulary (number of hash buckets), defaults to 2**15

    '''

    def __init__(self,
                 field_name: str,
                 vocab_size: int = 2 ** 15) -> None:
        super(BowFeaturizer, self).__init__(field_name, vocab_size)

        with mx.name.Prefix(field_name + "_"):
            self.symbol = mx.sym.Variable("{}".format(field_name), stype='csr')
