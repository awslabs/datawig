Parameters for Different Data Types
================================

This tutorial will highlight the different parameters associated with column data types supported by DataWig. We use the `SimpleImputer` in these examples, but the same concepts apply when using the `Imputer` and other encoders/featurizers.

The [python file](./params_tutorial.py)  in this directory contains the complete code for training models on text, numerical, and image data. Here, we illustrate examples of relevant parameters for training models on each of these types of data. 

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
    final_fc_hidden_units_candidates=[[100]],
    num_hash_bucket_candidates=[2**10, 2**15],
    tokens_candidates=['chars', 'words']
    )
```
Apart from the text parameters, `final_fc_hidden_units` corresponds to a list containing the dimensionality of the fully connected layer after all column features are concatenated. The length of this list is the number of hidden fully connected layers.

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
    hidden_layers_candidates=[0, 2],
    final_fc_hidden_units=[[100]]
    )
```
In this case, the model will use a fully connected layer size of 50 or 100, with 0 or 2 hidden layers.

## Image Data
When using images, the model expects the input data to have a column containing the path of the downloaded image for a particular sample. The model will feed the image through a pretrained network to extract features and then passes those features through fully connected layers.

The key parameter associated with image data is:

* `layer_dim` &mdash; list containing the dimensionality of the fully connected layer, where the length of the list is the number of hidden layers

Here is an example of using this parameter:

```python
imputer_image.fit_hpo(
    train_df=df_train,
    num_epochs=50,
    learning_rate_candidates=[1e-3, 1e-4],
    layer_dim=[[256], [1024, 512]],
    final_fc_hidden_units=[[100]]
)
```
In this case, the model will use one fully connected layer with size 256, or two fully connected layers with sizes of 1024 and then 512.
