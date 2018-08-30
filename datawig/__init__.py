# to avoid the caveats mentioned here such as non-deterministic execution we throw an exception
# when using chained assignment
# http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
import pandas as pd
pd.options.mode.chained_assignment = 'raise'

# makes the column encoders available as e.g. `from alster import CategoricalEncoder`
from .column_encoders import CategoricalEncoder, BowEncoder, NumericalEncoder, SequentialEncoder
from .mxnet_input_symbols import BowFeaturizer, LSTMFeaturizer, NumericalFeaturizer, EmbeddingFeaturizer
from .simple_imputer import SimpleImputer
from .imputer import Imputer

name = "datawig"