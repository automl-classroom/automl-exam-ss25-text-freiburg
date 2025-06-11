"""Text AutoML system for automatic text classification."""
from typing import Any, Tuple, Optional, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
from pathlib import Path
import logging

from .datasets import BaseTextDataset

logger = logging.getLogger(__name__)


class DummyTextClassifier(nn.Module):
    """Simple LSTM-based text classifier."""
    
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=128, output_size=2, max_length=512):
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # x is expected to be token ids of shape (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)  # Use last hidden state
        # Take the last hidden state
        output = self.classifier(hidden[-1])  # (batch_size, output_size)
        return output


class DummyBagOfWords(nn.Module):
    """Simple bag-of-words neural network classifier."""
    
    def __init__(self, vocab_size, output_size):
        super().__init__()
        hidden_size = 256
        self.model = nn.Sequential(
            nn.Linear(vocab_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        # x is expected to be bag-of-words vectors of shape (batch_size, vocab_size)
        return self.model(x)


class TextDataset(Dataset):
    """PyTorch Dataset for text data."""
    
    def __init__(self, texts, labels, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        if self.tokenizer:
            # Simple word-based tokenization (you can replace with more sophisticated tokenizers)
            tokens = text.split()[:self.max_length]
            token_ids = [self.tokenizer.get(token, 1) for token in tokens]  # 1 for unknown tokens
            
            # Pad or truncate
            if len(token_ids) < self.max_length:
                token_ids.extend([0] * (self.max_length - len(token_ids)))  # 0 for padding
            else:
                token_ids = token_ids[:self.max_length]
                
            return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)
        else:
            return text, torch.tensor(label, dtype=torch.long)


class TextAutoML:
    """Simple AutoML system for text classification."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.set_seed()
        
        # Model components
        self.model = None
        self.vectorizer = None
        self.tokenizer = None
        self.vocab_size = 10000
        self.max_length = 512
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.epochs = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data storage
        self.num_classes = None
        self.approach = "bag_of_words"  # Options: "bag_of_words", "tfidf", "neural"
        
    def set_seed(self):
        """Set random seeds for reproducibility."""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
    
    def fit(self, dataset_class):
        """Fit the AutoML system on the dataset."""
        logger.info("Loading and preparing data...")
        
        # Initialize dataset
        dataset = dataset_class()
        data_info = dataset.create_dataloaders(
            batch_size=self.batch_size, 
            test_size=0.2, 
            random_state=self.seed
        )
        
        train_df = data_info['train_df']
        val_df = data_info['val_df']
        self.num_classes = data_info['num_classes']
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df) if val_df is not None else 0}")
        logger.info(f"Number of classes: {self.num_classes}")
        
        # Choose approach based on data size (simple heuristic)
        if len(train_df) < 1000:
            self.approach = "bag_of_words"
        elif len(train_df) < 5000:
            self.approach = "tfidf" 
        else:
            self.approach = "neural"
            
        logger.info(f"Selected approach: {self.approach}")
        
        if self.approach in ["bag_of_words", "tfidf"]:
            self._fit_classical(train_df, val_df)
        else:
            self._fit_neural(train_df, val_df)
    
    def _fit_classical(self, train_df, val_df):
        """Fit classical ML approach with bag-of-words or TF-IDF."""
        if self.approach == "bag_of_words":
            self.vectorizer = CountVectorizer(
                max_features=self.vocab_size,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
        else:  # TF-IDF
            self.vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 2)
            )
        
        # Fit vectorizer and transform training data
        X_train = self.vectorizer.fit_transform(train_df['text'])
        y_train = train_df['label'].values
        
        # Create and train neural network with BoW features
        actual_vocab_size = X_train.shape[1]
        self.model = DummyBagOfWords(actual_vocab_size, self.num_classes)
        self.model.to(self.device)
        
        # Convert to dense tensors for training
        X_train_dense = torch.FloatTensor(X_train.toarray()).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Simple training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            # Simple batch processing
            for i in range(0, len(X_train_dense), self.batch_size):
                batch_x = X_train_dense[i:i+self.batch_size]
                batch_y = y_train_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item():.4f}")
    
    def _fit_neural(self, train_df, val_df):
        """Fit neural network approach with embeddings."""
        # Build vocabulary
        all_texts = train_df['text'].tolist()
        if val_df is not None:
            all_texts.extend(val_df['text'].tolist())
        
        # Simple tokenization to build vocabulary
        vocab = set()
        for text in all_texts:
            vocab.update(text.split())
        
        # Create word-to-index mapping
        vocab_list = ['<PAD>', '<UNK>'] + list(vocab)[:self.vocab_size-2]
        self.tokenizer = {word: i for i, word in enumerate(vocab_list)}
        actual_vocab_size = len(self.tokenizer)
        
        # Create model
        self.model = DummyTextClassifier(
            vocab_size=actual_vocab_size,
            embedding_dim=128,
            hidden_size=128,
            output_size=self.num_classes,
            max_length=self.max_length
        )
        self.model.to(self.device)
        
        # Create datasets and dataloaders
        train_dataset = TextDataset(
            train_df['text'], train_df['label'], 
            self.tokenizer, self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
    
    def predict(self, dataset_class) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on the test set."""
        logger.info("Making predictions...")
        
        # Load test data
        dataset = dataset_class()
        data_info = dataset.create_dataloaders(test_size=0.0)  # No validation split needed
        test_df = data_info['test_df']
        
        # Get test labels (if available)
        test_labels = test_df['label'].values if 'label' in test_df.columns else np.full(len(test_df), np.nan)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            if self.approach in ["bag_of_words", "tfidf"]:
                # Transform test data using fitted vectorizer
                X_test = self.vectorizer.transform(test_df['text'])
                X_test_dense = torch.FloatTensor(X_test.toarray()).to(self.device)
                
                # Predict in batches
                for i in range(0, len(X_test_dense), self.batch_size):
                    batch_x = X_test_dense[i:i+self.batch_size]
                    outputs = self.model(batch_x)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    predictions.extend(preds)
            
            else:  # Neural approach
                test_dataset = TextDataset(
                    test_df['text'], 
                    np.zeros(len(test_df)),  # Dummy labels
                    self.tokenizer, 
                    self.max_length
                )
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
                
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    outputs = self.model(batch_x)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    predictions.extend(preds)
        
        predictions = np.array(predictions)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions, test_labels