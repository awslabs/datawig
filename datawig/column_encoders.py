# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Column Encoders:
used for translating values of a table into numerical representation such that Featurizers can
operate on them
"""

import os
import random
from abc import ABCMeta, abstractmethod
from functools import partial
from typing import Any, Dict, Iterable, List

import mxnet as mx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .utils import logger

random.seed(0)
np.random.seed(42)


class NotFittedError(BaseException):
    """

    Error thrown when unfitted encoder is used

    """


class ColumnEncoder():
    """

    Abstract super class of column encoders.
    Transforms value representation of columns (e.g. strings) into numerical representations to be
    fed into MxNet.

    Options for ColumnEncoders are:

        SequentialEncoder:  for sequences of symbols (e.g. characters or words),
        BowEncoder: bag-of-word representation, as sparse vectors
        CategoricalEncoder: for categorical variables
        NumericalEncoder: for numerical values

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param output_dim: dimensionality of encoded column values (1 for categorical, vocabulary size
                    for sequential and BoW)

    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 input_columns: List[str],
                 output_column=None,
                 output_dim=1):

        if not isinstance(input_columns, list):
            input_columns = [input_columns]

        for col in input_columns:
            if not isinstance(col, str):
                raise ValueError("ColumnEncoder.input_columns must be str type, was {}".format(type(col)))

        if output_column is None:
            output_column = "-".join(input_columns)
            logstr = "No output column name provided for ColumnEncoder " \
                     "using {}".format(output_column)
            logger.debug(logstr)

        self.input_columns = input_columns
        self.output_column = output_column
        self.output_dim = int(output_dim)


    @abstractmethod
    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """

        Transforms values in one or more columns of DataFrame into a numpy array that can be fed
        into a Featurizer

        :param data_frame:
        :return: List of integers

        """
        pass

    @abstractmethod
    def fit(self, data_frame: pd.DataFrame):
        """

        Fits a ColumnEncoder if needed (i.e. vocabulary/alphabet)

        :param data_frame: pandas DataFrame
        :return:

        """
        return self

    @abstractmethod
    def is_fitted(self):
        """

        Checks if ColumnEncoder (still) needs to be fitted to data

        :return: True if the column encoder does not require fitting (anymore or at all)

        """
        pass

    @abstractmethod
    def decode(self, col: pd.Series) -> pd.Series:
        """

        Decodes a pandas Series of token indices

        :param col: pandas Series of token indices
        :return: pandas Series of tokens

        """
        pass


class CategoricalEncoder(ColumnEncoder):
    """

    Transforms categorical variable from string representation into number

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param token_to_idx: token to index mapping,
            0 is reserved for missing tokens, 1 ... max_tokens for most to least frequent tokens
    :param max_tokens: maximum number of tokens

    """

    def __init__(self,
                 input_columns: Any,
                 output_column: str = None,
                 token_to_idx: Dict[str, int] = None,
                 max_tokens: int = int(1e4)) -> None:

        ColumnEncoder.__init__(self, input_columns, output_column, 1)

        if len(self.input_columns) != 1:
            raise ValueError("CategoricalEncoder can only encode single columns, got {}: {}".format(
                len(self.input_columns), ", ".join(self.input_columns)))

        self.max_tokens = int(max_tokens)
        self.token_to_idx = token_to_idx
        self.idx_to_token = None

    @staticmethod
    def transform_func_categorical(col: pd.Series,
                                   token_to_idx: Dict[str, int],
                                   missing_token_idx: int) -> Any:
        """

        Transforms categorical values into their indices

        :param col: pandas Series with categorical values
        :param token_to_idx: Dict[str, int] with mapping from token to token index
        :param missing_token_idx: index for missing symbol
        :return:

        """
        return [token_to_idx.get(v, missing_token_idx) for v in col]

    def is_fitted(self):
        """

        Checks if ColumnEncoder (still) needs to be fitted to data

        :return: True if the column encoder does not require fitting (anymore or at all)

        """
        return self.token_to_idx is not None

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """

        Transforms string column of pandas dataframe into categoricals

        :param data_frame: pandas data frame
        :return: numpy array (rows by 1)

        """

        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if not self.token_to_idx:
            raise NotFittedError("CategoricalEncoder needs token to index mapping")

        if not self.idx_to_token:
            self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        func = partial(self.transform_func_categorical, token_to_idx=self.token_to_idx,
                       missing_token_idx=0)

        logger.debug("CategoricalEncoder encoding {} rows with \
                    {} tokens from column {} to \
                    column {}".format(len(data_frame), len(self.token_to_idx), self.input_columns[0], self.output_column))

        return data_frame[self.input_columns].apply(func).values.astype(np.float32)

    def fit(self, data_frame: pd.DataFrame):
        """

        Fits a CategoricalEncoder by extracting the value histogram of a column and capping it at
        max_tokens. Issues warning if less than 100 values were observed.

        :param data_frame: pandas data frame

        """

        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        value_histogram = data_frame[self.input_columns[0]].replace('',
                                                                    np.nan).dropna().value_counts()

        self.max_tokens = int(min(len(value_histogram), self.max_tokens))

        self.token_to_idx = {token: idx + 1 for idx, token in
                             enumerate(value_histogram.index[:self.max_tokens])}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        logger.debug("{} most often encountered discrete values: \
                     {}".format(self.max_tokens,value_histogram.index.values[:self.max_tokens]))

        for label in value_histogram.index[:self.max_tokens]:
            if value_histogram[label] < 100:
                logger.info("CategoricalEncoder for column {} \
                               found only {} occurrences of value {}".format(self.input_columns[0], value_histogram[label], label))

        return self

    def decode_token(self, token_idx: int) -> str:
        """

        Decodes a token index into a token

        :param token_idx: token index
        :return: token

        """
        return self.idx_to_token.get(token_idx, "MISSING")

    def decode(self, col: pd.Series) -> pd.Series:
        """

        Decodes a pandas Series of token indices

        :param col: pandas Series of token indices
        :return: pandas Series of tokens

        """
        return col.map(self.idx_to_token).fillna("MISSING")


class SequentialEncoder(ColumnEncoder):
    """

    Transforms sequence of characters into sequence of numbers

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param token_to_idx: token to index mapping
           0 is reserved for missing tokens, 1 ... max_tokens-1 for most to least frequent tokens
    :param max_tokens: maximum number of tokens
    :param seq_len: length of sequence, shorter sequences get padded to, longer sequences
                    truncated at seq_len symbols

    """

    def __init__(self,
                 input_columns: Any,
                 output_column: str = None,
                 token_to_idx: Dict[str, int] = None,
                 max_tokens: int = int(1e3),
                 seq_len: int = 500) -> None:

        ColumnEncoder.__init__(self, input_columns, output_column, seq_len)

        if len(self.input_columns) != 1:
            raise ValueError("SequentialEncoder can only encode single columns, got {}: {}".format(
                len(self.input_columns), ", ".join(self.input_columns)))

        self.token_to_idx = token_to_idx
        self.idx_to_token = None
        self.max_tokens = int(max_tokens)

    @staticmethod
    def transform_func_seq_single(string: str,
                                  token_to_idx: Dict[str, int],
                                  seq_len: int,
                                  missing_token_idx: int) -> List[int]:
        """

        Transforms a single string into a sequence of token ids

        :param string: a sequence of symbols as string
        :param token_to_idx: Dict[str, int] with mapping from token to token index
        :param seq_len: length of sequence
        :param missing_token_idx: index for missing symbol
        :return: List[int] with transformed values

        """
        if isinstance(string, str):
            rep = [token_to_idx.get(tok, missing_token_idx) for tok in string[:seq_len]]
        else:
            rep = []
        pad = [missing_token_idx] * (seq_len - len(rep))
        return rep + pad

    def is_fitted(self) -> bool:
        """

        Checks if ColumnEncoder (still) needs to be fitted to data

        :return: True if the column encoder does not require fitting (anymore or at all)

        """
        return self.token_to_idx is not None

    def fit(self, data_frame: pd.DataFrame):
        """

        Fits a SequentialEncoder by extracting the character value histogram of a column and
        capping it at max_tokens

        :param data_frame: pandas data frame

        """

        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        logger.debug("Fitting SequentialEncoder on %s rows", str(len(data_frame)))

        # get the relevant column and concatenate all strings
        flattened = pd.Series(
            list(data_frame[self.input_columns].astype(str).replace(' ', '').fillna(
                '').values.sum()))
        # compute character histograms, take most frequent ones
        char_hist = flattened.value_counts().sort_values(ascending=False)[:self.max_tokens]

        logstr = "Characters encoded for {}: {}".format(self.input_columns[0],
                                                        "".join(sorted(char_hist.index.values)))
        logger.debug(logstr)

        self.max_tokens = int(min(len(char_hist), self.max_tokens))
        self.token_to_idx = {token: idx + 1 for token, idx in
                             zip(char_hist.index, range(self.max_tokens))}
        self.idx_to_token = {idx: char for char, idx in self.token_to_idx.items()}

        return self

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """

        Transforms column of pandas dataframe into sequence of tokens

        :param data_frame: pandas DataFrame
        :return: numpy array (rows by seq_len)

        """

        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if not self.token_to_idx:
            raise NotFittedError("SequentialEncoder needs token to index mapping")

        logstr = "Applying SequentialEncoder on {} rows with {} tokens and seq_len {} " \
                 "from column {} to column {}".format(len(data_frame), len(self.token_to_idx),
                                                      self.output_dim, self.input_columns[0],
                                                      self.output_column)
        logger.debug(logstr)

        func = partial(self.transform_func_seq_single, token_to_idx=self.token_to_idx,
                       seq_len=self.output_dim,
                       missing_token_idx=0)

        return np.vstack(data_frame[self.input_columns[0]].apply(func).values).astype(np.float32)

    def decode_seq(self, token_index_sequence: Iterable[int]) -> str:
        """

        Decodes a sequence of token indices into a string

        :param token_index_sequence: an iterable of token indices
        :return: str the decoded string

        """
        return "".join([self.idx_to_token.get(token_idx, "") for token_idx in token_index_sequence])

    def decode(self, col: pd.Series) -> pd.Series:
        """
        Decodes a pandas Series of token indices

        :param col: pandas Series of token index iterables
        :return: pd.Series of strings

        """
        return col.apply(self.decode_seq).fillna("MISSING")


class TfIdfEncoder(ColumnEncoder):
    """

    TfIdf bag of word encoder for text data, using sklearn's TfidfVectorizer

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param max_tokens: Number of feature buckets (dimensionality of sparse ngram vector). default 2**18
    :param tokens: How to tokenize the input data, supports 'words' and 'chars'.
    :param ngram_range: length of ngrams to use as features
    :param prefixed_concatenation: whether or not to prefix values with column name before concat

    """

    def __init__(self,
                 input_columns: Any,
                 output_column: str = None,
                 max_tokens: int = 2 ** 18,
                 tokens: str = 'chars',
                 ngram_range: tuple = None,
                 prefixed_concatenation: bool = True) -> None:

        ColumnEncoder.__init__(self, input_columns, output_column, int(max_tokens))

        if ngram_range is None:
            ngram_range = (1, 3) if tokens == 'words' else (1, 5)

        if tokens == 'words':
            self.vectorizer = TfidfVectorizer(max_features=self.output_dim, ngram_range=ngram_range)
        elif tokens == 'chars':
            self.vectorizer = TfidfVectorizer(max_features=self.output_dim, ngram_range=ngram_range,
                                              analyzer="char")
        else:
            logger.debug(
                "BowEncoder attribute tokens has to be 'words' or 'chars', defaulting to 'chars'")
            self.vectorizer = TfidfVectorizer(max_features=self.output_dim,
                                              ngram_range=ngram_range, analyzer="char")

        self.prefixed_concatenation = prefixed_concatenation

        self.idx_to_token = None

        self.__fitted = False

    def __preprocess_input(self, data_frame: pd.DataFrame) -> pd.Series:
        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if len(self.input_columns) == 1:
            tmp_col = data_frame[self.input_columns[0]]
        else:
            tmp_col = pd.Series(index=data_frame.index, data='')
            for col in self.input_columns:
                if self.prefixed_concatenation:
                    tmp_col += col + " " + data_frame[col].fillna("") + " "
                else:
                    tmp_col += data_frame[col].fillna("") + " "
            logstr = "Applying TfIdf BoW Encoding to columns {} {} prefix into column {}".format(
                self.input_columns, "with" if self.prefixed_concatenation else "without",
                self.output_column)
            logger.debug(logstr)

        return tmp_col

    def fit(self, data_frame: pd.DataFrame):
        """
        :param data_frame:
        :return:

        """
        self.vectorizer.fit(self.__preprocess_input(data_frame))
        self.idx_to_token = {idx: token for token, idx in self.vectorizer.vocabulary_.items()}
        self.__fitted = True
        return self

    def is_fitted(self) -> bool:
        """
        :param self:
        :return: True if the encoder is fitted

        """
        return self.__fitted

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """
        Transforms one or more string columns into Bag-of-words vectors.

        :param data_frame: pandas DataFrame with text columns
        :return: numpy array (rows by max_features)

        """
        return self.vectorizer.transform(self.__preprocess_input(data_frame)).astype(np.float32)

    def decode(self, col: pd.Series) -> pd.Series:
        """

        Given a series of indices, decode it to input tokens

        :param col:
        :return: pd.Series of tokens

        """
        return col.map(lambda index: self.idx_to_token[index])


class BowEncoder(ColumnEncoder):
    """

    Bag-of-Words encoder for text data, using sklearn's HashingVectorizer

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param max_tokens: Number of hash buckets (dimensionality of sparse ngram vector). default 2**18
    :param tokens: How to tokenize the input data, supports 'words' and 'chars'.
    :param ngram_range: length of ngrams to use as features
    :param prefixed_concatenation: whether or not to prefix values with column name before concat

    """

    def __init__(self,
                 input_columns: Any,
                 output_column: str = None,
                 max_tokens: int = 2 ** 18,
                 tokens: str = 'chars',
                 ngram_range: tuple = None,
                 prefixed_concatenation: bool = True) -> None:

        if ngram_range is None:
            ngram_range = (1, 3) if tokens == 'words' else (1, 5)

        ColumnEncoder.__init__(self, input_columns, output_column, int(max_tokens))

        if tokens == 'words':
            self.vectorizer = HashingVectorizer(n_features=self.output_dim, ngram_range=ngram_range)
        elif tokens == 'chars':
            self.vectorizer = HashingVectorizer(n_features=self.output_dim, ngram_range=ngram_range,
                                                analyzer="char")
        else:
            logger.debug(
                "BowEncoder attribute tokens has to be 'words' or 'chars', defaulting to 'chars'")
            self.vectorizer = HashingVectorizer(n_features=self.output_dim, ngram_range=ngram_range,
                                                analyzer="char")

        self.prefixed_concatenation = prefixed_concatenation

    def fit(self, data_frame: pd.DataFrame):
        """

        Does nothing, HashingVectorizers do not need to be fit.

        :param data_frame:
        :return:

        """
        logger.debug("BowEncoder is stateless and doesn't need to be fit")
        return self

    def is_fitted(self) -> bool:
        """

        Returns true if the column encoder does not require fitting (anymore or at all)

        :param self:
        :return: True if the encoder is fitted

        """
        return True

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """

        Transforms one or more string columns into Bag-of-words vectors, hashed into a max_features
        dimensional feature space. Nans and missing values will be replaced by zero vectors.

        :param data_frame: pandas DataFrame with text columns
        :return: numpy array (rows by max_features)

        """
        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if len(self.input_columns) == 1:
            tmp_col = data_frame[self.input_columns[0]]
        else:
            tmp_col = pd.Series(index=data_frame.index, data='')
            for col in self.input_columns:
                if self.prefixed_concatenation:
                    tmp_col += col + " " + data_frame[col].fillna("") + " "
                else:
                    tmp_col += data_frame[col].fillna("") + " "
            logstr = "Applying Hashing BoW Encoding to columns {} {} prefix into column {}".format(
                self.input_columns, "with" if self.prefixed_concatenation else "without",
                self.output_column)
            logger.debug(logstr)

        return self.vectorizer.transform(tmp_col).astype(np.float32)

    def decode(self, col: pd.Series) -> pd.Series:
        """

        Raises NotImplementedError, hashed bag-of-words cannot be decoded due to hash collisions

        :param token_index_sequence:

        :return:

        """
        raise NotImplementedError


class NumericalEncoder(ColumnEncoder):
    """

    Numerical encoder, concatenates columns in field_names into one vector
    fills nans with the mean of a column

    :param input_columns: List[str] with column names to be used as input for this ColumnEncoder
    :param output_column: Name of output field, used as field name in downstream MxNet iterator
    :param normalize: whether to normalize by the standard deviation or not, default True

    """

    def __init__(self,
                 input_columns: Any,
                 output_column: str = None,
                 normalize=True) -> None:

        ColumnEncoder.__init__(self, input_columns, output_column, 0)

        self.output_dim = len(self.input_columns)
        self.normalize = normalize
        self.scaler = None

    def is_fitted(self):
        """

        Returns true if the column encoder does not require fitting (anymore or at all)

        :param self:
        :return: True if the encoder is fitted

        """
        fitted = True

        if self.normalize:
            fitted = self.scaler is not None

        return fitted

    def fit(self, data_frame: pd.DataFrame):
        """

        Does nothing or fits the normalizer, if normalization is specified

        :param data_frame: DataFrame with numerical columns specified when
                            instantiating NumericalEncoder

        """
        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        mean = data_frame[self.input_columns].mean()
        data_frame[self.input_columns] = data_frame[self.input_columns].fillna(mean)
        self.scaler = StandardScaler().fit(data_frame[self.input_columns].values)

        return self

    def transform(self, data_frame: pd.DataFrame) -> np.array:
        """

        Concatenates the numerical columns specified when instantiating the NumericalEncoder
        Normalizes features if specified in the NumericalEncoder

        :param data_frame: DataFrame with numerical columns specified in NumericalEncoder
        :return: np.array with numerical features (rows by number of numerical columns)

        """
        if not isinstance(data_frame, pd.core.frame.DataFrame):
            raise ValueError("Only pandas data frames are supported")

        if self.scaler is None:
            self.scaler = StandardScaler().fit(data_frame[self.input_columns].values)

        mean = pd.Series(dict(zip(self.input_columns,self.scaler.mean_)))
        data_frame[self.input_columns] = data_frame[self.input_columns].fillna(mean)

        logger.debug("Concatenating numeric columns %s into %s",
                    self.input_columns,
                    self.output_column)

        if self.normalize:
            logger.debug("Normalizing with StandardScaler")
            encoded = self.scaler.transform(data_frame[self.input_columns].values).astype(
                np.float32)
        else:
            encoded = data_frame[self.input_columns].values.astype(np.float32)
        return encoded

    def decode(self, col: pd.Series) -> pd.Series:
        """

        Undoes the normalization, scales by scale and adds the mean

        :param col: pandas Series (normalized)
        :return: pandas Series (unnormalized)

        """

        if self.normalize:
            decoded = (col * self.scaler.scale_) + self.scaler.mean_
        else:
            decoded = col

        return decoded
