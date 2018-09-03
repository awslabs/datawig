Introduction to `Imputer`
================================

This tutorial will teach you the basics of how to use the `Imputer` for your data imputation tasks. We will use a subset of the MAE data as an example. To download this data, please refer to [here](../README.md#data).

Open the [python file](./imputer_intro.py)  in this directory to see the code used in this tutorial.

## Load Data
First, let's load the data into a pandas DataFrame and split the data into train (80%) and test (20%) subsets.

 ```python
df = pd.read_csv('../finish_val_data_sample.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])
 ```

Note, the `random_split()` method is provided in `datawig.utils`. The validation set is partitioned from the train data during training and defaults to 10%.

## Default `Imputer`

The key difference with the `Imputer` is specifying the Encoders and Featurizers used for particular columns in your dataset. Once this is done, initializing the model, training, and making predictions with the Imputer is similar to the `SimpleImputer`

 ```python
 #Specify encoders and featurizers
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
 ```
 
For the input columns that contain data useful for imputation, the `Imputer` expects you to specify the particular encoders and featurizers. For the label column that your are trying to impute, only specifying the type of encoder is necessary.

## Using Different Encoders and Featurizers

One of the key advantages with the `Imputer` is that you get flexibility for customizing exactly which encoders and featurizers to use, which is something you can't do with the `SimpleImputer`. 

For example, let's say you wanted to use an LSTM rather than the default bag-of-words text model that the `SimpleImputer` uses. To do this, you can simply specificy the proper encoders and featurizers to initialize the `Imputer` model.

 ```python
 #Using LSTMs instead of bag-of-words
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
 ```
 
## Prediction with Probabilities
Beyond directly predicting values, the `Imputer` can also return the probabilities for each class on ever sample (numpy array of shape samples-by-labels). This can help with understanding what the model is predicting and with what probability for each sample.

```python
prob_dict = imputer.predict_proba(df_test)
```

In addition, you can get the probabilities only for the top-k most likely predicted classes (rather than for all the classes above).

```python
prob_dict_topk = imputer.predict_proba_top_k(df_test, top_k=5)
```

## Get Predictions and Metrics
To get predictions (original dataframe with an extra column) and the associated metrics from the validation set during training, you can run the following:

```python
predictions, metrics = imputer.transform_and_compute_metrics(df_test)
```