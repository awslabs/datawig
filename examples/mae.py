# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
from glob import glob
import pandas as pd
from collections import defaultdict
import requests, zipfile, io
import sys


if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'val']:
    print('Usage: python {} (train|val)'.format(sys.argv[0]))
    sys.exit(1)


def prepare_dataset(prefix):
    output_fn = "mae_{}_dataset.csv".format(prefix)

    def parse_and_store_dataset():
        df = []
        for f in glob('{}/*.json'.format(prefix)):
            print(f)
            j = json.load(open(f))
            for elem in j:
                elem['specs'] = defaultdict(lambda: None, elem['specs'])
                df.append(
                    {
                        'title': elem['title'],
                        'text': elem['text'],
                        'color': elem['specs']['color'],
                        'finish': elem['specs']['finish']
                    }
                )
        df = pd.DataFrame(df)

        df = df.loc[(~df.color.isna()) & (~df.finish.isna())]
        df.loc[:, 'color'] = df.loc[:, 'color'].str.replace(': ', '').str.lower()
        df.loc[:, 'finish'] = df.loc[:, 'finish'].str.replace(': ', '').str.lower()
        df.loc[:, 'text'] = df.loc[:, 'text'].str.replace('\n', ' ')
        df.loc[:, 'title'] = df.loc[:, 'title'].str.replace('\n', ' ')

        n_distinct_categorical = 16

        df = df.loc[(df.color.isin(df.color.value_counts().index[:n_distinct_categorical])) & (df.finish.isin(df.finish.value_counts().index[:n_distinct_categorical]))]
        df[['color', 'finish', 'title', 'text']].to_csv(output_fn, index=False)

    fn = "https://s3-us-west-2.amazonaws.com/mumie/{}.zip".format(prefix)
    print('Starting download of {}, this could take a while...'.format(fn))
    r = requests.get(fn)
    z = zipfile.ZipFile(io.BytesIO(r.content))

    print('Unzipping {}'.format(fn))
    z.extractall()

    print('Parsing MAE dataset and storing it in {}'.format(output_fn))
    parse_and_store_dataset()


prepare_dataset(sys.argv[1])
