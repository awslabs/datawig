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

from datawig import Imputer
from datawig.column_encoders import *
from datawig.mxnet_input_symbols import *
from datawig.utils import random_split
import pandas as pd

"""
Load Data
"""
df = pd.read_csv('mae_train_dataset.csv').sample(n=1000)
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

# ------------------------------------------------------------------------------------

"""
Run default Imputer
"""
data_encoder_cols = [BowEncoder('title'),
                     BowEncoder('text')]
label_encoder_cols = [CategoricalEncoder('finish')]
data_featurizer_cols = [BowFeaturizer('title'),
                        BowFeaturizer('text')]

imputer = Imputer(
    data_featurizers=data_featurizer_cols,
    label_encoders=label_encoder_cols,
    data_encoders=data_encoder_cols,
    output_path='imputer_model'
)

imputer.fit(train_df=df_train)
predictions = imputer.predict(df_test)

# ------------------------------------------------------------------------------------

"""
Specifying Encoders and Featurizers
"""
data_encoder_cols = [SequentialEncoder('title'),
                     SequentialEncoder('text')]
label_encoder_cols = [CategoricalEncoder('finish')]
data_featurizer_cols = [LSTMFeaturizer('title'),
                        LSTMFeaturizer('text')]

imputer = Imputer(
    data_featurizers=data_featurizer_cols,
    label_encoders=label_encoder_cols,
    data_encoders=data_encoder_cols,
    output_path='imputer_model'
)

imputer.fit(train_df=df_train, num_epochs=5)
predictions = imputer.predict(df_test)

# ------------------------------------------------------------------------------------

"""
Run Imputer with predict_proba/predict_proba_top_k
"""
prob_dict = imputer.predict_proba(df_test)
prob_dict_topk = imputer.predict_proba_top_k(df_test, top_k=5)

# ------------------------------------------------------------------------------------

"""
Run Imputer with transform_and_compute_metrics
"""
predictions, metrics = imputer.transform_and_compute_metrics(df_test)
