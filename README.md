DataWig - Imputation for Tables
================================

[![PyPI version](https://badge.fury.io/py/datawig.svg)](https://badge.fury.io/py/datawig.svg)
[![GitHub license](https://img.shields.io/github/license/awslabs/datawig.svg)](https://github.com/awslabs/datawig/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/datawig.svg)](https://github.com/awslabs/datawig/issues)
[![Build Status](https://travis-ci.org/awslabs/datawig.svg?branch=master)](https://travis-ci.org/awslabs/datawig)

DataWig learns Machine Learning models to impute missing values in tables.

See our user-guide and extended documentation [here](https://datawig.readthedocs.io/en/latest).

## Installation

### CPU
```bash
pip3 install datawig
```

### GPU
If you want to run DataWig on a GPU you need to make sure your version of Apache MXNet Incubating contains the GPU bindings.
Depending on your version of CUDA, you can do this by running the following:

```bash
wget https://raw.githubusercontent.com/awslabs/datawig/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
pip install datawig --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).

## Running DataWig
The DataWig API expects your data as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). Here is an example of how the dataframe might look:

|Product Type | Description           | Size | Color |
|-------------|-----------------------|------|-------|
|   Shoe      | Ideal for Running     | 12UK | Black |
| SDCards     | Best SDCard ever ...  | 8GB  | Blue  |
| Dress       | This **yellow** dress | M    | **?** |

### Quickstart Example

For most use cases, the `SimpleImputer` class is the best starting point. For convenience there is the function [SimpleImputer.complete](https://datawig.readthedocs.io/en/latest/source/API.html#datawig.simple_imputer.SimpleImputer.complete) that takes a DataFrame and fits an imputation model for each column with missing values, with all other columns as inputs:

```python
import datawig, numpy

# generate some data with simple nonlinear dependency
df = datawig.utils.generate_df_numeric() 
# mask 10% of the values
df_with_missing = df.mask(numpy.random.rand(*df.shape) > .9)

# impute missing values
df_with_missing_imputed = datawig.SimpleImputer.complete(df_with_missing)

```

You can also impute values in specific columns only (called `output_column` below) using values in other columns (called `input_columns` below). DataWig currently supports imputation of categorical columns and numeric columns.

### Imputation of categorical columns

```python
import datawig

df = datawig.utils.generate_df_string( num_samples=200, 
                                       data_column_name='sentences', 
                                       label_column_name='label')

df_train, df_test = datawig.utils.random_split(df)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['sentences'], # column(s) containing information about the column we want to impute
    output_column='label', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)
```

### Imputation of numerical columns

```python
import datawig

df = datawig.utils.generate_df_numeric( num_samples=200, 
                                        data_column_name='x', 
                                        label_column_name='y')         
df_train, df_test = datawig.utils.random_split(df)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    input_columns=['x'], # column(s) containing information about the column we want to impute
    output_column='y', # the column we'd like to impute values for
    output_path = 'imputer_model' # stores model data and metrics
    )

#Fit an imputer model on the train data
imputer.fit(train_df=df_train, num_epochs=50)

#Impute missing values and return original dataframe with predictions
imputed = imputer.predict(df_test)
             
```

In order to have more control over the types of models and preprocessings, the `Imputer` class allows directly specifying all relevant model features and parameters. 

For details on usage, refer to the provided [examples](./examples).

### Acknowledgments
Thanks to [David Greenberg](https://github.com/dgreenberg) for the package name.

### Building documentation

```bash
git clone git@github.com:awslabs/datawig.git
cd datawig/docs
make html
open _build/html/index.html
```


### Executing Tests

Clone the repository from git and set up virtualenv in the root dir of the package:

```
python3 -m venv venv
```

Install the package from local sources:

```
./venv/bin/pip install -e .
```

Run tests:

```
./venv/bin/pip install -r requirements/requirements.dev.txt
./venv/bin/python -m pytest
```


### Updating PyPi distribution

Before updating, increment the version in setup.py.

```
git clone git@github.com:awslabs/datawig.git
cd datawig
# build local distribution for current version
python setup.py sdist
# upload to PyPi
twine upload --skip-existing dist/*
```

