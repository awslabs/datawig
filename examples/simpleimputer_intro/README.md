Introduction to `SimpleImputer`
================================

This tutorial will teach you the basics of how to use `SimpleImputer` for your data imputation tasks. We will use a subset of the MAE data as an example. To download this data, please refer to [here](../README.md#data).

Open the [python file](./simpleimputer_intro.py) in this directory to see the code used in this tutorial.

## Load Data
First, let's load the data into a pandas DataFrame and split the data into train (80%) and test (20%) subsets.

 ```python
df = pd.read_csv('../finish_mae_dataset.csv')
df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])
 ```

Note, the `random_split()` method is provided in `datawig.utils`. The validation set is partitioned from the train data during training and defaults to 10%.

## Default `SimpleImputer`

At the most basic level, you can run the `SimpleImputer` on data without specifying any additional arguments. This will automatically choose the right `ColumnEncoder` and `Featurizer` for each column and train an imputation model with default hyperparameters.

To train a model, you can simply initialize a `SimpleImputer`, specifying the input columns containing useful data for imputation, the output column that you'd like to impute values for, and the output path, which will store model data and metrics. Then, you can use the `fit()` method to train the model.

 ```python
#Initialize a SimpleImputer model
imputer = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path = 'imputer_model'
    )

#Fit an imputer model on the train data 
imputer.fit(train_df=df_train)
 ```
 
 From here, you can this model to make predictions on the test set and return the original dataframe with an additional column containing the model's predictions.
 
 ```python
predictions = imputer.predict(df_test)
 ```
 
 Finally, you can determine useful metrics to gauge how well the model's predictions compare to the true values (using `sklearn.metrics`).
 
 ```python
#Calculate f1 score
f1 = f1_score(predictions['finish'], predictions['finish_imputed']) 

#Print overall classification report
print(classification_report(predictions['finish'], predictions['finish_imputed']))
 ```

## `SimpleImputer` with HPO

DataWig also enables hyperparameter optimization to find the best model on a particular dataset.

The steps for training a model with HPO are identical to the default `SimpleImputer`.

```python
imputer = SimpleImputer(
    input_columns=['title', 'text'],
    output_column='finish',
    output_path = 'imputer_model'
    )

#Fit an imputer model with customized hyperparameters
imputer.fit_hpo(
        train_df=df_train,
        num_epochs=100,
        patience=3,
        learning_rate_candidates=[1e-3, 3e-4, 1e-4],
        hpo_max_train_samples=1000
    	)
```
See the [`SimpleImputer` code](https://github.com/awslabs/datawig/blob/97e259d6fde9e38f66c59e82a068172c54060c04/datawig/simple_imputer.py#L144-L162) for more details on parameters.

We also have a tutorial [here](../params_tutorial) that covers more details on relevant parameters for text, numerical, and image data.

## Load Saved Model
Once a model is trained, it will be saved in the location of `output_path`, which you specified as an argument when intializing the `SimpleImputer`. You can easily load this model for further experiments or run on new datasets as follows.

```python
#Load saved model
imputer = SimpleImputer.load('./imputer_model')
```
This model also contains the associated metrics (stored as a dictionary) calculated on the validation set during training.

```python
#Load metrics from the validation set
metrics = imputer.load_metrics()
weighted_f1 = metrics['weighted_f1']
avg_precision = metrics['avg_precision']
...
``` 

