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
import math

# Try to import transformers for advanced models
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from .datasets import BaseTextDataset

logger = logging.getLogger(__name__)


class DummyTextClassifier(nn.Module):
    """Simple accuracy_score
    M-based text classifier."""
    
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


class DummyTransformerClassifier(nn.Module):
    """Simple encoder-only transformer for text classification."""
    
    def __init__(self, vocab_size, embed_dim=128, num_heads=8, num_layers=4, 
                 hidden_dim=512, output_size=2, max_length=512, dropout=0.1):
        super().__init__()
        self.max_length = max_length
        self.embed_dim = embed_dim
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_size)
        )
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(x)  # (batch_size, seq_length, embed_dim)
        pos_embeds = self.position_embedding(positions)
        embeddings = self.layer_norm(token_embeds + pos_embeds)
        embeddings = self.dropout(embeddings)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (x != 0)  # Assuming 0 is padding token
        
        # Transformer expects inverted mask (True for positions to mask)
        src_key_padding_mask = ~attention_mask
        
        # Apply transformer
        transformer_output = self.transformer(embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Global average pooling (ignoring padded positions)
        mask = attention_mask.unsqueeze(-1).float()  # (batch_size, seq_length, 1)
        pooled = (transformer_output * mask).sum(dim=1) / mask.sum(dim=1)  # (batch_size, embed_dim)
        
        # Classification
        output = self.classifier(pooled)
        return output


class PretrainedClassifier(nn.Module):
    """Wrapper for fine-tuning pretrained models."""
    
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for pretrained models")
            
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


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


class PretrainedTextDataset(Dataset):
    """PyTorch Dataset for pretrained model tokenization."""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx])
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        # Tokenize using the pretrained tokenizer
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


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
        self.approach = "bag_of_words"  # Options: "bag_of_words", "tfidf", "neural-lstm", "neural-transformer", "finetune"
        
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
        
        # Choose approach based on data size and available resources
        if len(train_df) < 500:
            self.approach = "bag_of_words"
        elif len(train_df) < 2000:
            self.approach = "tfidf" 
        elif len(train_df) < 10000:
            self.approach = "neural-lstm"
        elif len(train_df) < 20000 and TRANSFORMERS_AVAILABLE:
            self.approach = "finetune"  # Use pretrained for medium-large datasets
        elif TRANSFORMERS_AVAILABLE:
            self.approach = "neural-transformer"
        else:
            self.approach = "neural-lstm"  # Fallback if transformers not available
            
        logger.info(f"Selected approach: {self.approach}")
        
        if self.approach in ["bag_of_words", "tfidf"]:
            self._fit_classical(train_df, val_df)
        elif self.approach == "neural-lstm":
            self._fit_neural_lstm(train_df, val_df)
        elif self.approach == "neural-transformer":
            self._fit_neural_transformer(train_df, val_df)
        elif self.approach == "finetune":
            self._fit_pretrained(train_df, val_df)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
    
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
    
    def _fit_neural_lstm(self, train_df, val_df):
        """Fit LSTM neural network approach with embeddings."""
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
    
    def _fit_neural_transformer(self, train_df, val_df):
        """Fit transformer neural network approach."""
        # Build vocabulary (same as LSTM)
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
        
        # Create transformer model
        self.model = DummyTransformerClassifier(
            vocab_size=actual_vocab_size,
            embed_dim=128,
            num_heads=8,
            num_layers=4,
            hidden_dim=256,
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
        
        # Training (similar to LSTM but with different learning rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate * 0.5)  # Lower LR for transformer
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
                
                # Gradient clipping for transformer stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 2 == 0:  # More frequent logging for transformer
                logger.info(f"Epoch {epoch+1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")
    
    def _fit_pretrained(self, train_df, val_df):
        """Fine-tune a pretrained model."""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("transformers library not available, falling back to LSTM")
            self._fit_neural_lstm(train_df, val_df)
            return
        
        # Choose a lightweight pretrained model suitable for laptops
        model_names = [
            "distilbert-base-uncased",  # Fastest, good for laptops
            "distilroberta-base",       # Alternative
            "albert-base-v2"            # Very small
        ]
        
        model_name = model_names[0]  # Default to DistilBERT
        logger.info(f"Using pretrained model: {model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = PretrainedClassifier(model_name, self.num_classes)
            self.model.to(self.device)
            
            # Create datasets
            train_dataset = PretrainedTextDataset(
                train_df['text'], train_df['label'], 
                self.tokenizer, max_length=min(self.max_length, 256)  # Shorter for speed
            )
            
            val_dataset = None
            if val_df is not None:
                val_dataset = PretrainedTextDataset(
                    val_df['text'], val_df['label'],
                    self.tokenizer, max_length=min(self.max_length, 256)
                )
            
            # Training arguments optimized for laptops
            training_args = TrainingArguments(
                output_dir='./temp_trainer',
                num_train_epochs=3,  # Fewer epochs for speed
                per_device_train_batch_size=8,  # Small batch size for memory
                per_device_eval_batch_size=16,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir='./temp_logs',
                logging_steps=50,
                evaluation_strategy="steps" if val_dataset else "no",
                eval_steps=100 if val_dataset else None,
                save_strategy="no",  # Don't save checkpoints to save space
                load_best_model_at_end=False,
                fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
                dataloader_num_workers=0,  # Avoid multiprocessing issues
                remove_unused_columns=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if val_dataset else None
            )
            
            # Train
            logger.info("Starting fine-tuning...")
            trainer.train()
            logger.info("Fine-tuning completed!")
            
        except Exception as e:
            logger.error(f"Error in fine-tuning: {e}")
            logger.info("Falling back to LSTM approach...")
            self.approach = "neural-lstm"
            self._fit_neural_lstm(train_df, val_df)
    
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
            if self.approach in ["bag_of_words"]:  #, "tfidf"]:
                # Transform test data using fitted vectorizer
                X_test = self.vectorizer.transform(test_df['text'])
                X_test_dense = torch.FloatTensor(X_test.toarray()).to(self.device)
                
                # Predict in batches
                for i in range(0, len(X_test_dense), self.batch_size):
                    batch_x = X_test_dense[i:i+self.batch_size]
                    outputs = self.model(batch_x)
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    predictions.extend(preds)
            
            elif self.approach in ["neural-lstm"]:  #, "neural-transformer"]:
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
            
            elif self.approach == "finetune":
                # Use the pretrained tokenizer and model
                test_dataset = PretrainedTextDataset(
                    test_df['text'],
                    np.zeros(len(test_df)),  # Dummy labels
                    self.tokenizer,
                    max_length=min(self.max_length, 256)
                )
                test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
                
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    predictions.extend(preds)
        
        predictions = np.array(predictions)
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions, test_labels