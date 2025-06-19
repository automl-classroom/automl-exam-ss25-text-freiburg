"""Text AutoML package for automatic text classification."""

from .automl import (
    TextAutoML, 
    DummyTextClassifier, 
    DummyBagOfWords, 
    DummyTransformerClassifier,
    PretrainedClassifier,
    TextDataset,
    PretrainedTextDataset
)
from .datasets import BaseTextDataset, AGNewsDataset, IMDBDataset, AmazonReviewsDataset

__all__ = [
    'TextAutoML',
    'DummyTextClassifier', 
    'DummyBagOfWords',
    'DummyTransformerClassifier',
    'PretrainedClassifier',
    'TextDataset',
    'PretrainedTextDataset',
    'BaseTextDataset',
    'AGNewsDataset',
    'IMDBDataset', 
    'AmazonReviewsDataset'
]