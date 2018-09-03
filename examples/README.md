Step-by-step Examples
================================

## Setup
For installing DataWig, follow the [installation instructions](../README.md).

## Examples
Below is a list of usage examples for DataWig. In each example, we provide a detailed description of important features along with python code that highlights these features on a public dataset. We recommend reading through the [overview of DataWig](#overview-of-datawig) and then following the below examples in order.

1. [Introduction to `SimpleImputer`](./simpleimputer_intro)
2. [Introduction to `Imputer`](./imputer_intro)
3. [Parameters for Different Data Types](./params_tutorial)

For additional examples and use cases, refer to the [unit test cases](https://github.com/awslabs/datawig/blob/master/test/test_imputer.py#L278).

## Data
Unless otherwise specified, these examples will make use of the [Multimodal Attribute Extraction (MAE) dataset](https://arxiv.org/pdf/1711.11118.pdf). This dataset contains over 2.2 million products with corresponding attributes, but to make data loading and processing more manageable, we provide a reformatted subset of the validation data (for the *finish* attribute) as a .csv file. 

This data contains columns for *title*, *text*, and *finish*. The title and text columns contain string data that will be used to impute the finish attribute. Note, the dataset is extremely noisy, but still provides a good example for real-world use cases of DataWig.

Run the following in this directory to download the dataset:

```bash
> wget https://s3.amazonaws.com/datawig/example_data/finish_val_data.csv
```

If you'd like to use this data in your own experiments, please remember to cite the original MAE paper:

```
@article{RobertLLogan2017MultimodalAE,
  title={Multimodal Attribute Extraction},
  author={IV RobertL.Logan and Samuel Humeau and Sameer Singh},
  journal={CoRR},
  year={2017},
  volume={abs/1711.11118}
}
```

## Overview of DataWig
Here, we give a brief overview of the internals of DataWig.

###ColumnEncoder (*column_encoder.py*)

Defines an abstract super class of column encoders that transforms the raw data of a column (e.g. strings from a product title) into an encoded numerical representation.

There are a few options for ColumnEncoders (subclasses) depending on the column data type:

* `SequentialEncoder`&mdash;  for sequences of string symbols (e.g. characters or words)
* `BowEncoder`&mdash; bag-of-word representation for strings, as sparse vectors
* `CategoricalEncoder`&mdash; for categorical variables (one-hot encoding)
* `NumericalEncoder`&mdash; for numerical values
* `ImageEncoder`&mdash; for processing images

###Featurizer (*mxnet_input\_symbol.py*)

Defines a specific featurizer for data that has been encoded into a numerical format by ColumnEncoder. The Featurizer is used to feed data into the imputation model's computational graph for training and prediction.

There are a few options for Featurizers depending on which ColumnEncoder was used for a particular column:

* `LSTMFeaturizer`&mdash; maps an input representing a sequence of symbols into a latent vector using an LSTM
* `BowFeaturizer`&mdash; used with `BowEncoder` on string data
* `EmbeddingFeaturizer`&mdash; maps encoded catagorical data into a vector representations (word-embeddings)
* `NumericalFeaturizer`&mdash; extracts features from numerical data using fully connected layers
* `ImageFeaturizer`&mdash; extracts image features using a standard CNN network architecture

###SimpleImputer (*simple_imputer.py*)
Using `SimpleImputer` is the easiest way to deploy an imputation model on your dataset with DataWig. As the name suggests, the `SimpleImputer` is straightforward to call from a python script and uses default encoders and featurizers that usually yield good results on a variety of datasets.

###Imputer (*imputer.py*)
`Imputer` is the backbone of the `SimpleImputer` and is responsible for running the preprocessing code, creating the model, executing training, and making predictions. Using the `Imputer` enables more flexibility with specifying model parameters, such as using particular encoders and featurizers rather than the default ones that `SimpleImputer` uses.