DataWig - Imputation for Tables
================================

[![PyPI version](https://badge.fury.io/py/datawig.svg)](https://badge.fury.io/py/datawig.svg)
[![GitHub license](https://img.shields.io/github/license/awslabs/datawig.svg)](https://github.com/awslabs/datawig/blob/master/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/awslabs/datawig.svg)](https://github.com/awslabs/datawig/issues)
[![Build Status](https://travis-ci.org/awslabs/datawig.svg?branch=master)](https://travis-ci.org/awslabs/datawig)

DataWig learns models to impute missing values in tables.

For each to-be-imputed column, DataWig trains a supervised machine learning model
to predict the observed values in that column using the data from other columns.

## Dependencies

DataWig requires:
- **Python3**
- MXNet 1.3.0
- numpy
- pandas
- scikit-learn

## Installation with pip
### CPU
```bash
> pip3 install datawig
```

### GPU
If you want to run DataWig on a GPU you need to make sure your version of Apache MXNet Incubating contains the GPU bindings.
Depending on your version of CUDA, you can do this by running the following:

```bash
> wget https://raw.githubusercontent.com/awslabs/datawig/master/requirements/requirements.gpu-cu${CUDA_VERSION}.txt
> pip install datawig --no-deps -r requirements.gpu-cu${CUDA_VERSION}.txt
> rm requirements.gpu-cu${CUDA_VERSION}.txt
```
where `${CUDA_VERSION}` can be `75` (7.5), `80` (8.0), `90` (9.0), or `91` (9.1).

## Running DataWig
The DataWig API expects your data as a [pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). Here is an example of how the dataframe might look:

![datawig dataframe example](https://s3.amazonaws.com/datawig/example_data/df_image_resize.png)


For most use cases, the `SimpleImputer` class is the best starting point. DataWig expects you to provide the column name of the column you would like to impute values for (called `output_column` below) and some column names that contain values that you deem useful for imputation (called `input_columns` below).

 ```python
    from datawig import SimpleImputer
    import pandas as pd

    df_train = pd.read_csv('/path/to/train/data.csv')
    df_test = pd.read_csv('/path/to/test/data.csv')

    #Initialize a SimpleImputer model
    imputer = SimpleImputer(
        input_columns=['item_name', 'description'], #columns containing information about the column we want to impute
        output_column='brand', #the column we'd like to impute values for
        output_path = 'imputer_model' #stores model data and metrics
        )
    
    #Fit an imputer model on the train data
    imputer.fit(train_df=df_train)

    #Impute missing values and return original dataframe with predictions
    imputed = imputer.predict(df_test)
 ```

In order to have more control over the types of models and preprocessings, the `Imputer` class allows directly specifying all relevant model features and parameters. 

For details on usage, refer to the provided [examples](./examples).

## Executing Tests

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

### Acknowledgments
Thanks to [David Greenberg](https://github.com/dgreenberg) for the package name.
