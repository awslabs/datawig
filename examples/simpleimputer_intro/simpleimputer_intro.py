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

from datawig import SimpleImputer
from datawig.utils import random_split
from sklearn.metrics import f1_score, classification_report
import pandas as pd

'''
Load Data
'''
df = pd.read_csv('../finish_val_data_sample.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

#------------------------------------------------------------------------------------

'''
Run default SimpleImputer
'''
#Initialize a SimpleImputer model
imputer = SimpleImputer(
    input_columns=['title', 'text'], #columns containing information about the column we want to impute
    output_column='finish', #the column we'd like to impute values for
    output_path = 'imputer_model' #stores model data and metrics
    )

#Fit an imputer model on the train data 
imputer.fit(train_df=df_train)

#Impute missing values and return original dataframe with predictions
predictions = imputer.predict(df_test)

#Calculate f1 score for true vs predicted values
f1 = f1_score(predictions['finish'], predictions['finish_imputed']) 

#Print overall classification report
print(classification_report(predictions['finish'], predictions['finish_imputed']))

#------------------------------------------------------------------------------------

'''
Run SimpleImputer with hyperparameter optimization
'''
#Initialize a SimpleImputer model
imputer = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path = 'imputer_model'
    )

#Fit an imputer model with default list of hyperparameters
imputer.fit_hpo(train_df=df_train)

#Fit an imputer model with customized HPO
imputer.fit_hpo(
        train_df=df_train,
        num_epochs=100,
        patience=3,
        learning_rate_candidates=[1e-3, 3e-4, 1e-4],
        hpo_max_train_samples=1000
    	)

#------------------------------------------------------------------------------------

'''
Load saved model and get metrics from SimpleImputer
'''
#Load saved model
imputer = SimpleImputer.load('./imputer_model')

#Load a dictionary of metrics from the validation set
metrics = imputer.load_metrics()
weighted_f1 = metrics['weighted_f1']
avg_precision = metrics['avg_precision']
#... explore other metrics stored in this dictionary!
