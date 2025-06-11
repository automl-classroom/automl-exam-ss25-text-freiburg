"""Text AutoML package for automatic text classification."""

from .automl import TextAutoML, DummyTextClassifier, DummyBagOfWords, TextDataset
from .datasets import BaseTextDataset, AGNewsDataset, IMDBDataset, AmazonReviewsDataset

__all__ = [
    'TextAutoML',
    'DummyTextClassifier', 
    'DummyBagOfWords',
    'TextDataset',
    'BaseTextDataset',
    'AGNewsDataset',
    'IMDBDataset', 
    'AmazonReviewsDataset'
]