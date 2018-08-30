DataWig - Imputation for Tables
================================

DataWig learns models to impute missing values in tables. 

For each to-be-imputed column, DataWig trains a supervised machine learning model
to predict the observed values in that column from the values in other columns  

# Installation with pip

```
pip install datawig
```

# Running Tests

Set up virtualenv in the root dir of the package:

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
