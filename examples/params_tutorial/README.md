Parameters for Different Data Types
================================

This tutorial will highlight the different parameters associated with column data types supported by DataWig. We use the `SimpleImputer` in these examples, but the same concepts apply when using the `Imputer` and other encoders/featurizers.

The [python file](./params_tutorial.py)  in this directory contains the complete code for training models on text and numerical data. Here, we illustrate examples of relevant parameters for training models on each of these types of data. 

It's important to note that your dataset can contain columns with mixed types. The `SimpleImputer` automatically determines which encoder and featurizer to use when training an imputation model!

## Text Data
The key parameters associated with text data are:

* `num_hash_buckets` &mdash; dimensionality of the vector for bag-of-words
* `tokens` &mdash; type of tokenization used for text data (default: chars)

Here is an example of using these parameters:

```python
imputer_text.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    num_hash_bucket_candidates=[2**10, 2**15],
    tokens_candidates=['chars', 'words']
    )
```

## Numerical Data
The key parameters associated with numerical data are:

* `latent_dim` &mdash; dimensionality of the fully connected layers for creating a feature vector from numerical data
* `hidden_layers` &mdash; number of fully connected layers

Here is an example of using these parameters:

```python
imputer_numeric.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    latent_dim_candidates=[50, 100],
    hidden_layers_candidates=[0, 2]
    )
```
In this case, the model will use a fully connected layer size of 50 or 100, with 0 or 2 hidden layers.