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
from io import BytesIO

from PIL import Image


def save_image_file(filename, color='red'):
    if color == 'red':
        color_vector = (155, 0, 0)
    elif color == 'green':
        color_vector = (0, 155, 0)
    elif color == 'blue':
        color_vector = (0, 0, 155)
    file = BytesIO()
    image = Image.new('RGBA', size=(50, 50), color=color_vector)
    image.save(filename, 'png')
    file.name = filename + '.png'
    file.seek(0)

    return file
