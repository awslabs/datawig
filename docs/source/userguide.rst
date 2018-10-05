User Guide
==========

Step-by-step Examples
---------------------

Setup
*****

For installing DataWig, follow the `installation instructions in the readme`_.

Examples
********

In each example, we provide a detailed description of important features along with python code that highlights these features on a public dataset. We recommend reading through the overview of DataWig and then following the below examples in order.

For additional examples and use cases, refer to the `unit test cases`_.

Data
****
Unless otherwise specified, these examples will make use of the `Multimodal Attribute Extraction (MAE) dataset`_. This dataset contains over 2.2 million products with corresponding attributes, but to make data loading and processing more manageable, we provide a reformatted subset of the validation data (for the *finish* attribute) as a .csv file.

This data contains columns for *title*, *text*, and *finish*. The title and text columns contain string data that will be used to impute the finish attribute. Note, the dataset is extremely noisy, but still provides a good example for real-world use cases of DataWig.

To speed up run-time, all examples will use a smaller version of this finish dataset that contains only 500 samples. Run the following in this directory to download this dataset:

.. code-block:: bash

    wget https://www.dropbox.com/s/nw6jf4inin83tb9/finish_val_data_sample.csv


To get the complete finish dataset with all data, run the following (we highly recommend running with a gpu to speed up computation):

.. code-block:: bash

    wget https://www.dropbox.com/s/1scdnb60kvilkuf/finish_val_data_all.csv


If you'd like to use this data in your own experiments, please remember to cite the original MAE paper:

.. code-block:: bibtex

    @article{RobertLLogan2017MultimodalAE,
      title={Multimodal Attribute Extraction},
      author={IV RobertL.Logan and Samuel Humeau and Sameer Singh},
      journal={CoRR},
      year={2017},
      volume={abs/1711.11118}


Overview of DataWig
*******************

Here, we give a brief overview of the internals of DataWig.

ColumnEncoder (*column_encoder.py*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines an abstract super class of column encoders that transforms the raw data of a column (e.g. strings from a product title) into an encoded numerical representation.

There are a few options for ColumnEncoders (subclasses) depending on the column data type:

* :code:`SequentialEncoder`:  for sequences of string symbols (e.g. characters or words)
* :code:`BowEncoder`: bag-of-word representation for strings, as sparse vectors
* :code:`CategoricalEncoder`: for categorical variables (one-hot encoding)
* :code:`NumericalEncoder`: for numerical values
* :code:`ImageEncoder`: for processing images

Featurizer (*mxnet_input\_symbol.py*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Defines a specific featurizer for data that has been encoded into a numerical format by ColumnEncoder. The Featurizer is used to feed data into the imputation model's computational graph for training and prediction.

There are a few options for Featurizers depending on which ColumnEncoder was used for a particular column:

* :code:`LSTMFeaturizer` maps an input representing a sequence of symbols into a latent vector using an LSTM
* :code:`BowFeaturizer` used with :code:`BowEncoder` on string data
* :code:`EmbeddingFeaturizer` maps encoded catagorical data into a vector representations (word-embeddings)
* :code:`NumericalFeaturizer` extracts features from numerical data using fully connected layers
* :code:`ImageFeaturizer` extracts image features using a standard CNN network architecture

SimpleImputer (*simple_imputer.py*)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using :code:`SimpleImputer` is the easiest way to deploy an imputation model on your dataset with DataWig. As the name suggests, the :code:`SimpleImputer` is straightforward to call from a python script and uses default encoders and featurizers that usually yield good results on a variety of datasets.

Imputer (*imputer.py*)
^^^^^^^^^^^^^^^^^^^^^^

:code:`Imputer` is the backbone of the :code:`SimpleImputer` and is responsible for running the preprocessing code, creating the model, executing training, and making predictions. Using the :code:`Imputer` enables more flexibility with specifying model parameters, such as using particular encoders and featurizers rather than the default ones that :code:`SimpleImputer` uses.


Introduction to Imputer
-----------------------

This tutorial will teach you the basics of how to use the :code:`Imputer` for your data imputation tasks. We will use a subset of the MAE data as an example. To download this data, please refer to README_.

Open `Imputer intro`_ to see the code used in this tutorial.

Load Data
*********

First, let's load the data into a pandas DataFrame and split the data into train (80%) and test (20%) subsets.

.. code-block:: python

    df = pd.read_csv('../finish_val_data_sample.csv')
    df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])

Note, the :code:`random_split()` method is provided in :code:`datawig.utils`. The validation set is partitioned from the train data during training and defaults to 10%.

Default :code:`Imputer`
***********************

The key difference with the :code:`Imputer` is specifying the Encoders and Featurizers used for particular columns in your dataset. Once this is done, initializing the model, training, and making predictions with the Imputer is similar to the :code:`SimpleImputer`

.. code-block:: python

    #Specify encoders and featurizers
    data_encoder_cols = [BowEncoder('title'), BowEncoder('text')]
    label_encoder_cols = [CategoricalEncoder('finish')]
    data_featurizer_cols = [BowFeaturizer('title'), BowFeaturizer('text')]

    imputer = Imputer(
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path='imputer_model'
    )

    imputer.fit(train_df=df_train)
    predictions = imputer.predict(df_test)


For the input columns that contain data useful for imputation, the :code:`Imputer` expects you to specify the particular encoders and featurizers. For the label column that your are trying to impute, only specifying the type of encoder is necessary.

Using Different Encoders and Featurizers
****************************************

One of the key advantages with the :code:`Imputer` is that you get flexibility for customizing exactly which encoders and featurizers to use, which is something you can't do with the :code:`SimpleImputer`.

For example, let's say you wanted to use an LSTM rather than the default bag-of-words text model that the :code:`SimpleImputer` uses. To do this, you can simply specificy the proper encoders and featurizers to initialize the :code:`Imputer` model.

.. code-block:: python

    #Using LSTMs instead of bag-of-words
    data_encoder_cols = [SequentialEncoder('title'), SequentialEncoder('text')]
    label_encoder_cols = [CategoricalEncoder('finish')]
    data_featurizer_cols = [LSTMFeaturizer('title'), LSTMFeaturizer('text')]

    imputer = Imputer(
        data_featurizers=data_featurizer_cols,
        label_encoders=label_encoder_cols,
        data_encoders=data_encoder_cols,
        output_path='imputer_model'
    )

Prediction with Probabilities
*****************************

Beyond directly predicting values, the :code:`Imputer` can also return the probabilities for each class on ever sample (numpy array of shape samples-by-labels). This can help with understanding what the model is predicting and with what probability for each sample.

.. code-block:: python

    prob_dict = imputer.predict_proba(df_test)

In addition, you can get the probabilities only for the top-k most likely predicted classes (rather than for all the classes above).

.. code-block:: python

    prob_dict_topk = imputer.predict_proba_top_k(df_test, top_k=5)


Get Predictions and Metrics
***************************
To get predictions (original dataframe with an extra column) and the associated metrics from the validation set during training, you can run the following:

.. code-block:: python

    predictions, metrics = imputer.transform_and_compute_metrics(df_test)


Introduction to :code:`SimpleImputer`
-------------------------------------

This tutorial will teach you the basics of how to use :code:`SimpleImputer` for your data imputation tasks. We will use a subset of the MAE data as an example. To download this data, please refer to the previous section.

Open the `SimpleImputer intro`_ in this directory to see the code used in this tutorial.

Load Data
*********
First, let's load the data into a pandas DataFrame and split the data into train (80%) and test (20%) subsets.

.. code-block:: python

    df = pd.read_csv('../finish_val_data_sample.csv')
    df_train, df_test = random_split(df, split_ratios=[0.8, 0.2])


Note, the :code:`random_split()` method is provided in :code:`datawig.utils`. The validation set is partitioned from the train data during training and defaults to 10%.

Default :code:`SimpleImputer`
*****************************

At the most basic level, you can run the :code:`SimpleImputer` on data without specifying any additional arguments. This will automatically choose the right :code:`ColumnEncoder` and :code:`Featurizer` for each column and train an imputation model with default hyperparameters.

To train a model, you can simply initialize a :code:`SimpleImputer`, specifying the input columns containing useful data for imputation, the output column that you'd like to impute values for, and the output path, which will store model data and metrics. Then, you can use the :code:`fit()` method to train the model.

.. code-block:: python

    #Initialize a SimpleImputer model
    imputer = SimpleImputer(
        input_columns=['title', 'text'],
        output_column='finish',
        output_path = 'imputer_model'
    )

    #Fit an imputer model on the train data
    imputer.fit(train_df=df_train)


From here, you can this model to make predictions on the test set and return the original dataframe with an additional column containing the model's predictions.

.. code-block:: python

    predictions = imputer.predict(df_test)

Finally, you can determine useful metrics to gauge how well the model's predictions compare to the true values (using :code:`sklearn.metrics`).

.. code-block:: python

    #Calculate f1 score
    f1 = f1_score(predictions['finish'], predictions['finish_imputed'])

    #Print overall classification report
    print(classification_report(predictions['finish'], predictions['finish_imputed']))

HPO with :code:`SimpleImputer`
******************************

DataWig also enables hyperparameter optimization to find the best model on a particular dataset.

The steps for training a model with HPO are identical to the default :code:`SimpleImputer`.

.. code-block:: python

    imputer = SimpleImputer(
        input_columns=['title', 'text'],
        output_column='finish',
        output_path='imputer_model'
    )

    #Fit an imputer model with customized hyperparameters
    imputer.fit_hpo(
        train_df=df_train,
        num_epochs=100,
        patience=3,
        learning_rate_candidates=[1e-3, 3e-4, 1e-4],
        hpo_max_train_samples=1000
    )

See the SimpleImputer_ for more details on parameters.

We also have a tutorial that covers more details on relevant parameters for text, numerical, and image data.

Load Saved Model
****************

Once a model is trained, it will be saved in the location of :code:`output_path`, which you specified as an argument when intializing the :code:`SimpleImputer`. You can easily load this model for further experiments or run on new datasets as follows.

.. code-block:: python

    #Load saved model
    imputer = SimpleImputer.load('./imputer_model')

This model also contains the associated metrics (stored as a dictionary) calculated on the validation set during training.

.. code-block:: python

    #Load metrics from the validation set
    metrics = imputer.load_metrics()
    weighted_f1 = metrics['weighted_f1']
    avg_precision = metrics['avg_precision']
    # ...


Parameters for Different Data Types
-----------------------------------

This tutorial will highlight the different parameters associated with column data types supported by DataWig. We use the :code:`SimpleImputer` in these examples, but the same concepts apply when using the :code:`Imputer` and other encoders/featurizers.

The `parameter tutorial`_ contains the complete code for training models on text, numerical, and image data. Here, we illustrate examples of relevant parameters for training models on each of these types of data.

It's important to note that your dataset can contain columns with mixed types. The :code:`SimpleImputer` automatically determines which encoder and featurizer to use when training an imputation model!

Text Data
*********

The key parameters associated with text data are:

* :code:`num_hash_buckets`  dimensionality of the vector for bag-of-words
* :code:`tokens`  type of tokenization used for text data (default: chars)

Here is an example of using these parameters:

.. code-block:: python

    imputer_text.fit_hpo(
        train_df=df_train,
        num_epochs=50,
        learning_rate_candidates=[1e-3, 1e-4],
        final_fc_hidden_units_candidates=[[100]],
        num_hash_bucket_candidates=[2**10, 2**15],
        tokens_candidates=['chars', 'words']
    )

Apart from the text parameters, :code:`final_fc_hidden_units` corresponds to a list containing the dimensionality of the fully connected layer after all column features are concatenated. The length of this list is the number of hidden fully connected layers.

Numerical Data
**************

The key parameters associated with numerical data are:

* :code:`latent_dim`  dimensionality of the fully connected layers for creating a feature vector from numerical data
* :code:`hidden_layers`  number of fully connected layers

Here is an example of using these parameters:

.. code-block:: python

    imputer_numeric.fit_hpo(
        train_df=df_train,
        num_epochs=50,
        learning_rate_candidates=[1e-3, 1e-4],
        latent_dim_candidates=[50, 100],
        hidden_layers_candidates=[0, 2],
        final_fc_hidden_units=[[100]]
    )

In this case, the model will use a fully connected layer size of 50 or 100, with 0 or 2 hidden layers.

Image Data
**********

When using images, the model expects the input data to have a column containing the path of the downloaded image for a particular sample. The model will feed the image through a pretrained network to extract features and then passes those features through fully connected layers.

The key parameter associated with image data is:

* :code:`layer_dim`  list containing the dimensionality of the fully connected layer, where the length of the list is the number of hidden layers

Here is an example of using this parameter:

.. code-block:: python

    imputer_image.fit_hpo(
        train_df=df_train,
        num_epochs=50,
        learning_rate_candidates=[1e-3, 1e-4],
        layer_dim=[[256], [1024, 512]],
        final_fc_hidden_units=[[100]]
    )

In this case, the model will use one fully connected layer with size 256, or two fully connected layers with sizes of 1024 and then 512.


.. _README: https://github.com/awslabs/datawig/blob/master/README.md
.. _`installation instructions in the readme`: https://github.com/awslabs/datawig/blob/master/README.md
.. _`unit test cases`: https://github.com/awslabs/datawig/blob/master/test/test_imputer.py#L278
.. _`Multimodal Attribute Extraction (MAE) dataset`: https://arxiv.org/pdf/1711.11118.pdf
.. _`Imputer intro`: https://github.com/awslabs/datawig/blob/master/examples/imputer_intro/imputer_intro.py
.. _`SimpleImputer intro`: https://github.com/awslabs/datawig/blob/master/examples/simpleimputer_intro/simpleimputer_intro.py
.. _SimpleImputer: https://github.com/awslabs/datawig/blob/97e259d6fde9e38f66c59e82a068172c54060c04/datawig/simple_imputer.py#L144-L162
.. _`parameter tutorial`: https://github.com/awslabs/datawig/blob/master/examples/params_tutorial/params_tutorial.py
