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

DataWig imputer output modules

"""

from typing import List, Tuple, Any
import mxnet as mx
from .utils import logger


def make_categorical_loss(latents: mx.symbol,
                          label_field_name: str,
                          num_labels: int,
                          final_fc_hidden_units: List[int] = None) -> Tuple[Any, Any]:
    '''

    Generate output symbol for categorical loss

    :param latents: MxNet symbol containing the concantenated latents from all featurizers
    :param label_field_name: name of the label column
    :num_labels: number of labels contained in the label column (for prediction)
    :final_fc_hidden_units: list of dimensions for the final fully connected layer.
                            The length of this list corresponds to the number of FC 
                            layers, and the contents of the list are integers with
                            corresponding hidden layer size.
    :return: mxnet symbols for predictions and loss

    '''

    if not final_fc_hidden_units:
        # generate prediction symbol
        fully_connected = mx.sym.FullyConnected(
            data=latents,
            num_hidden=num_labels,
            name="label_{}".format(label_field_name))
    else:
        layer_size = final_fc_hidden_units
        with mx.name.Prefix("label_{}".format(label_field_name)):
            for i, layer in enumerate(layer_size):
                if i == len(layer_size) - 1:
                    fully_connected = mx.sym.FullyConnected(
                        data=latents,
                        num_hidden=layer)
                else:
                    latents = mx.sym.FullyConnected(
                        data=latents,
                        num_hidden=layer)

    pred = mx.sym.softmax(fully_connected)
    label = mx.sym.Variable(label_field_name)

    # assign to 0.0 the label values larger than number of classes so that they
    # do not contribute to the loss

    logger.info("Building output of label {} with {} classes \
                 (including missing class)".format(label, num_labels))

    num_labels_vec = label * 0.0 + num_labels
    indices = mx.sym.broadcast_lesser(label, num_labels_vec)
    label = label * indices

    # goes from (batch, 1) to (batch,) as it is required for softmax output
    label = mx.sym.split(label, axis=1, num_outputs=1, squeeze_axis=1)

    # mask entries when label is 0 (missing value)
    missing_labels = mx.sym.zeros_like(label)
    positive_mask = mx.sym.broadcast_greater(label, missing_labels)

    # compute the cross entropy only when labels are positive
    cross_entropy = mx.sym.pick(mx.sym.log_softmax(fully_connected), label) * -positive_mask

    # normalize the cross entropy by the number of positive label
    num_positive_indices = mx.sym.sum(positive_mask)
    cross_entropy = mx.sym.broadcast_div(cross_entropy, num_positive_indices + 1.0)

    # todo because MakeLoss normalize even with normalization='null' argument is used,
    # we have to multiply by batch_size here
    batch_size = mx.sym.sum(mx.sym.ones_like(label))
    cross_entropy = mx.sym.broadcast_mul(cross_entropy, batch_size)

    return pred, cross_entropy


def make_numerical_loss(latents: mx.symbol, label_field_name: str) -> Tuple[Any, Any]:
    '''

    Generate output symbol for univariate numeric loss

    :param label_field_name:
    :return: mxnet symbols for predictions and loss

    '''

    # generate prediction symbol
    pred = mx.sym.FullyConnected(
        data=latents,
        num_hidden=1,
        name="label_{}".format(label_field_name))

    target = mx.sym.Variable(label_field_name)

    # squared loss
    loss = mx.sym.sum((pred - target) ** 2.0)

    return pred, loss
