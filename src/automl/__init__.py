# """Text AutoML package for automatic text classification."""

from .core import (
    TextAutoML,
    SimpleFFNN,
    LSTMClassifier,
    # Uncomment the following lines if you have these classes implemented 
    # DummyTextClassifier, 
    # DummyBagOfWords, 
    # DummyTransformerClassifier,
    # PretrainedClassifier,
    # TextDataset,
    SimpleTextDataset,
    # PretrainedTextDataset
)
from .datasets import (
    AGNewsDataset,
    IMDBDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    YelpDataset,
)


__all__ = [
    'TextAutoML',
    'SimpleFFNN',
    'LSTMClassifier',
    'SimpleTextDataset',
    'AGNewsDataset',
    'IMDBDataset', 
    'AmazonReviewsDataset',
    'DBpediaDataset',
    'YelpDataset',
]
# end of file