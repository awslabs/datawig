from datawig import Imputer
from datawig.column_encoders import *
from datawig.mxnet_input_symbols import *
from datawig.utils import random_split
import pandas as pd

'''
Load Data
'''
df = pd.read_csv('../finish_val_data.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#------------------------------------------------------------------------------------

'''
Run default Imputer
'''
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

#------------------------------------------------------------------------------------

'''
Specifying Encoders and Featurizers
'''
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

imputer.fit(train_df=df_train)
predictions = imputer.predict(df_test)

#------------------------------------------------------------------------------------

'''
Run Imputer with predict_proba/predict_proba_top_k
'''
prob_dict = imputer.predict_proba(df_test)
prob_dict_topk = imputer.predict_proba_top_k(df_test, top_k=5)

#------------------------------------------------------------------------------------

'''
Run Imputer with transform_and_compute_metrics
'''
predictions, metrics = imputer.transform_and_compute_metrics(df_test)

