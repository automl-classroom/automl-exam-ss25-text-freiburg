"""Dataset classes for NLP AutoML tasks."""
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import string
from sklearn.model_selection import train_test_split


class BaseTextDataset(ABC):
    """Base class for text datasets."""
    
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or Path("data")
        self.vocab_size = 10000  # Default vocab size
        self.max_length = 512    # Default max sequence length
        
    @abstractmethod
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test data."""
        pass
    
    @abstractmethod
    def get_num_classes(self) -> int:
        """Return number of classes."""
        pass
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def create_dataloaders(self, batch_size: int = 32, test_size: float = 0.2, 
                          random_state: int = 42) -> Dict[str, Any]:
        """Create train/validation/test dataloaders and preprocessing objects."""
        train_df, test_df = self.load_data()
        
        # Split training data into train/validation
        if test_size > 0:
            train_df, val_df = train_test_split(
                train_df, test_size=test_size, random_state=random_state, 
                stratify=train_df['label'] if 'label' in train_df.columns else None
            )
        else:
            val_df = None
        
        # Preprocess text
        train_df['text'] = train_df['text'].apply(self.preprocess_text)
        if val_df is not None:
            val_df['text'] = val_df['text'].apply(self.preprocess_text)
        test_df['text'] = test_df['text'].apply(self.preprocess_text)
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'num_classes': self.get_num_classes()
        }


class AGNewsDataset(BaseTextDataset):
    """AG News dataset for news categorization (4 classes)."""
    
    def get_num_classes(self) -> int:
        return 4
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load AG News data."""
        # This assumes CSV files with columns: label, text
        train_path = self.data_path / "ag_news" / "train.csv"
        test_path = self.data_path / "ag_news" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            # Generate dummy data for demonstration
            print(f"Data files not found at {train_path}, generating dummy data...")
            train_df = self._generate_dummy_data(1000, is_train=True)
            test_df = self._generate_dummy_data(200, is_train=False)
        
        return train_df, test_df
    
    def _generate_dummy_data(self, n_samples: int, is_train: bool = True) -> pd.DataFrame:
        """Generate dummy AG News data for testing."""
        categories = ["World", "Sports", "Business", "Technology"]
        templates = {
            0: ["Global news about {}", "International event in {}", "World update on {}"],
            1: ["Sports match between {} and {}", "{} wins championship", "Athletic competition in {}"],
            2: ["Company {} reports earnings", "Business deal with {}", "Market update on {}"],
            3: ["New technology {}", "Tech company {} launches", "Innovation in {}"]
        }
        
        np.random.seed(42 if is_train else 24)
        labels = np.random.randint(0, 4, n_samples)
        texts = []
        
        for label in labels:
            template = np.random.choice(templates[label])
            if "{}" in template:
                if template.count("{}") == 1:
                    text = template.format(f"entity_{np.random.randint(1, 100)}")
                else:
                    text = template.format(f"team_{np.random.randint(1, 50)}", 
                                         f"team_{np.random.randint(51, 100)}")
            else:
                text = template
            texts.append(text + " " + " ".join([f"word_{np.random.randint(1, 1000)}" 
                                              for _ in range(np.random.randint(10, 50))]))
        
        return pd.DataFrame({'text': texts, 'label': labels})


class IMDBDataset(BaseTextDataset):
    """IMDB movie review sentiment dataset (2 classes)."""
    
    def get_num_classes(self) -> int:
        return 2
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load IMDB data."""
        train_path = self.data_path / "imdb" / "train.csv"
        test_path = self.data_path / "imdb" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            print(f"Data files not found at {train_path}, generating dummy data...")
            train_df = self._generate_dummy_data(2000, is_train=True)
            test_df = self._generate_dummy_data(400, is_train=False)
        
        return train_df, test_df
    
    def _generate_dummy_data(self, n_samples: int, is_train: bool = True) -> pd.DataFrame:
        """Generate dummy IMDB data for testing."""
        positive_words = ["excellent", "amazing", "great", "fantastic", "wonderful", "brilliant"]
        negative_words = ["terrible", "awful", "horrible", "bad", "disappointing", "boring"]
        
        np.random.seed(42 if is_train else 24)
        labels = np.random.randint(0, 2, n_samples)
        texts = []
        
        for label in labels:
            if label == 1:  # Positive
                sentiment_words = np.random.choice(positive_words, 3)
                text = f"This movie was {sentiment_words[0]} and {sentiment_words[1]}. Really {sentiment_words[2]}!"
            else:  # Negative
                sentiment_words = np.random.choice(negative_words, 3)
                text = f"This movie was {sentiment_words[0]} and {sentiment_words[1]}. Really {sentiment_words[2]}!"
            
            # Add random filler text
            text += " " + " ".join([f"word_{np.random.randint(1, 1000)}" 
                                  for _ in range(np.random.randint(20, 100))])
            texts.append(text)
        
        return pd.DataFrame({'text': texts, 'label': labels})


class AmazonReviewsDataset(BaseTextDataset):
    """Amazon product reviews dataset (5 classes for categories)."""
    
    def get_num_classes(self) -> int:
        return 5
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load Amazon reviews data."""
        train_path = self.data_path / "amazon" / "train.csv"
        test_path = self.data_path / "amazon" / "test.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        else:
            print(f"Data files not found at {train_path}, generating dummy data...")
            train_df = self._generate_dummy_data(3000, is_train=True)
            test_df = self._generate_dummy_data(600, is_train=False)
        
        return train_df, test_df
    
    def _generate_dummy_data(self, n_samples: int, is_train: bool = True) -> pd.DataFrame:
        """Generate dummy Amazon reviews data for testing."""
        categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
        templates = {
            0: ["This electronic device is {}", "The gadget works {}", "Electronics quality is {}"],
            1: ["This book is {}", "Reading experience was {}", "The story is {}"],
            2: ["This clothing item is {}", "The fabric feels {}", "Fashion choice is {}"],
            3: ["This home item is {}", "For the house, it's {}", "Home utility is {}"],
            4: ["This sports equipment is {}", "For exercise, it's {}", "Athletic gear is {}"]
        }
        
        quality_words = ["good", "excellent", "poor", "average", "outstanding"]
        
        np.random.seed(42 if is_train else 24)
        labels = np.random.randint(0, 5, n_samples)
        texts = []
        
        for label in labels:
            template = np.random.choice(templates[label])
            quality = np.random.choice(quality_words)
            text = template.format(quality)
            
            # Add random review text
            text += " " + " ".join([f"word_{np.random.randint(1, 1000)}" 
                                  for _ in range(np.random.randint(15, 80))])
            texts.append(text)
        
        return pd.DataFrame({'text': texts, 'label': labels})