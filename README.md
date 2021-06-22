DataWig - Imputation for Tables
================================

[![GitHub license](https://img.shields.io/github/license/awslabs/datawig.svg)](https://github.com/awslabs/datawig/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/datawig.svg)](https://github.com/awslabs/datawig/issues)

DataWig learns Machine Learning models to impute missing values in tables.

The latest version of DataWig is built around the [tabular prediction API of AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html).

This change will lead to better imputation models and faster training -- but not all of the original DataWig API is yet migrated.

## Installation

Clone the repository from git and set up virtualenv in the root dir of the package:

```
python3 -m venv venv
```

Install the package from local sources:

```
./venv/bin/pip install -e .
```

## Running DataWig
The DataWig API expects your data as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). Here is an example of how the dataframe might look:

|Product Type | Description           | Size | Color |
|-------------|-----------------------|------|-------|
|   Shoe      | Ideal for Running     | 12UK | Black |
| SDCards     | Best SDCard ever ...  | 8GB  | Blue  |
| Dress       | This **yellow** dress | M    | **?** |

DataWig let's you impute missing values in two ways:
  * A `.complete` functionality inspired by [`fancyimpute`](https://github.com/iskandr/fancyimpute)
  * A `sklearn`-like API with `.fit` and `.predict` methods

## Quickstart Example

Here are some examples of the DataWig API, also available as [notebook](datawig-examples.ipynb)

### Using `AutoGluonImputer.complete`

```python
import datawig, numpy

# generate some data with simple nonlinear dependency
df = datawig.utils.generate_df_numeric()
# mask 10% of the values
df_with_missing = df.mask(numpy.random.rand(*df.shape) > .9)

# impute missing values
df_with_missing_imputed = datawig.AutoGluonImputer.complete(df_with_missing)

```

### Using `AutoGluonImputer.fit` and `.predict`

This usage is very similar to using the underlying [tabular prediction API of AutoGluon](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html) - but we added some convenience functionality such as a precision filtering for categorical imputations.  

You can also impute values in specific columns only (called `output_column` below) using values in other columns (called `input_columns` below). DataWig currently supports imputation of categorical columns and numeric columns. Type inference is based on [``pandas.api.types.is_numeric_dtype``](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_numeric_dtype.html) .

#### Imputation of categorical columns

Let's first generate some random strings hidden in longer random strings:

```python
import datawig

df = datawig.utils.generate_df_string( num_samples=200,
                                       data_column_name='sentences',
                                       label_column_name='label')
df.head(n=2)
```

The generate data will look like this:

|sentences	|label|
|---------|-------|
|	wILsn T366D r1Psz KAnDn 8RfUf GuuRU	|8RfUf|
|	8RfUf jBq5U BqVnh pnXfL GuuRU XYnSP	|8RfUf|

Now let's split the rows into training and test data and train an imputation model

```python
df_train, df_test = datawig.utils.random_split(df)

imputer = datawig.AutoGluonImputer(
    input_columns=['sentences'], # column(s) containing information about the column we want to impute
    output_column='label' # the column we'd like to impute values for
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, time_limit=100)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)
```

#### Imputation of numerical columns

Imputation of numerical values works just like for categorical values.

Let's first generate some numeric values with a quadratic dependency:

```python
import datawig

df = datawig.utils.generate_df_numeric( num_samples=200,
                                        data_column_name='x',
                                        label_column_name='y')      

df_train, df_test = datawig.utils.random_split(df)

imputer = datawig.AutoGluonImputer(
    input_columns=['x'], # column(s) containing information about the column we want to impute
    output_column='y', # the column we'd like to impute values for
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, time_limit=100)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)
```


### Acknowledgments
Thanks to [David Greenberg](https://github.com/dgreenberg) for the package name.


### Executing Tests

Run tests:

```
./venv/bin/pip install -r requirements/requirements.dev.txt
./venv/bin/python -m pytest
```
