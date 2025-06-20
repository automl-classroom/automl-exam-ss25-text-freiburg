"""Text AutoML package for automatic text classification."""

from .automl import (
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
from .datasets import BaseTextDataset, AGNewsDataset, IMDBDataset, AmazonReviewsDataset

__all__ = [
    'TextAutoML',
    'SimpleFFNN',
    'LSTMClassifier',
    # 'DummyTextClassifier', 
    # 'DummyBagOfWords',
    # 'DummyTransformerClassifier',
    # 'PretrainedClassifier',
    # 'TextDataset',
    # 'PretrainedTextDataset',
    # 'BaseTextDataset',
    'SimpleTextDataset',
    'AGNewsDataset',
    'IMDBDataset', 
    'AmazonReviewsDataset'
]