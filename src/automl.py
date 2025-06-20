import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from src.models import SimpleFFNN, LSTMClassifier
from src.utils import SimpleTextDataset
import logging
from typing import Tuple
from collections import Counter
from neps import NePS

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAutoML:
    def __init__(self, seed=42, approach='auto', vocab_size=10000, token_length=128, epochs=5,
                 batch_size=64, lr=1e-4, weight_decay=0.0,
                 ffnn_hidden=128, lstm_emb_dim=128, lstm_hidden_dim=128):
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.approach = approach
        self.vocab_size = vocab_size
        self.token_length = token_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        self.ffnn_hidden = ffnn_hidden
        self.lstm_emb_dim = lstm_emb_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.num_classes = None
        self.val_texts = []
        self.val_labels = []

    def fit(self, train_df, val_df, num_classes, seed=42, approach=None, vocab_size=None, token_length=None,
            epochs=None, batch_size=None, lr=None, weight_decay=None,
            ffnn_hidden=None, lstm_emb_dim=None, lstm_hidden_dim=None):
        """
        Fits a model to the given dataset.

        Parameters:
        - train_df (pd.DataFrame): Training data with 'text' and 'label' columns.
        - val_df (pd.DataFrame): Validation data with 'text' and 'label' columns.
        - num_classes (int): Number of classes in the dataset.
        - seed (int): Random seed for reproducibility.
        - approach (str): Model type - 'tfidf', 'lstm', or 'transformer'. Default is 'auto'.
        - vocab_size (int): Maximum vocabulary size.
        - token_length (int): Maximum token sequence length.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - lr (float): Learning rate.
        - weight_decay (float): Weight decay for optimizer.
        - ffnn_hidden (int): Hidden dimension size for FFNN.
        - lstm_emb_dim (int): Embedding dimension size for LSTM.
        - lstm_hidden_dim (int): Hidden dimension size for LSTM.
        """
        if approach is not None: self.approach = approach
        if vocab_size is not None: self.vocab_size = vocab_size
        if token_length is not None: self.token_length = token_length
        if epochs is not None: self.epochs = epochs
        if batch_size is not None: self.batch_size = batch_size
        if lr is not None: self.lr = lr
        if weight_decay is not None: self.weight_decay = weight_decay
        if ffnn_hidden is not None: self.ffnn_hidden = ffnn_hidden
        if lstm_emb_dim is not None: self.lstm_emb_dim = lstm_emb_dim
        if lstm_hidden_dim is not None: self.lstm_hidden_dim = lstm_hidden_dim

        logger.info("Loading and preparing data...")

        train_texts = train_df['text'].tolist()
        train_labels = train_df['label'].tolist()
        self.val_texts = val_df['text'].tolist()
        self.val_labels = val_df['label'].tolist()
        self.num_classes = num_classes
        logger.info(f"Train class distribution: {Counter(train_labels)}")
        logger.info(f"Val class distribution: {Counter(self.val_labels)}")

        if self.approach == 'auto':
            n = len(train_texts)
            if n < 500:
                self.approach = 'tfidf'
            elif n < 2000:
                self.approach = 'lstm'
            elif TRANSFORMERS_AVAILABLE:
                self.approach = 'transformer'
            else:
                self.approach = 'lstm'

        if self.approach == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.vocab_size)
            X = self.vectorizer.fit_transform(train_texts).toarray()
            dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(train_labels))
            self.model = SimpleFFNN(X.shape[1], hidden=self.ffnn_hidden, output_dim=self.num_classes)

        elif self.approach == 'lstm':
            vocab = set(w for t in train_texts for w in t.split())
            vocab = ['<PAD>', '<UNK>'] + list(vocab)[:self.vocab_size - 2]
            self.tokenizer = {w: i for i, w in enumerate(vocab)}
            dataset = SimpleTextDataset(train_texts, train_labels, self.tokenizer, self.token_length)
            self.model = LSTMClassifier(len(self.tokenizer), self.lstm_emb_dim, self.lstm_hidden_dim, self.num_classes)

        elif self.approach == 'transformer' and TRANSFORMERS_AVAILABLE:
            model_name = 'distilbert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            dataset = SimpleTextDataset(train_texts, train_labels, self.tokenizer, self.token_length)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_classes)

        else:
            raise ValueError("Unsupported approach or missing transformers.")

        self.model.to(self.device)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_acc = self._train_loop(loader)
        return val_acc

    def _train_loop(self, loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                self.model.train()
                optimizer.zero_grad()
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    logits = outputs.logits
                    labels = inputs['labels']
                else:
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    outputs = self.model(x)
                    labels = y
                    loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.info(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

            if self.val_texts:
                val_preds = self.predict_from_texts(self.val_texts)
                val_acc = accuracy_score(self.val_labels, val_preds)
                logger.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")

        if self.val_texts:
            val_preds = self.predict_from_texts(self.val_texts)
            val_acc = accuracy_score(self.val_labels, val_preds)
            logger.info(f"Epoch {epoch + 1}, Validation Accuracy: {val_acc:.4f}")
        
        return val_acc or 0.0


    def predict_from_texts(self, texts):
        self.model.eval()
        preds = []

        if self.vectorizer:
            X = self.vectorizer.transform(texts).toarray()
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model(X_tensor)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
        else:
            dataset = SimpleTextDataset(texts, [0] * len(texts), self.tokenizer, self.token_length)
            loader = DataLoader(dataset, batch_size=16)
            with torch.no_grad():
                for batch in loader:
                    if isinstance(batch, dict):
                        inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                        outputs = self.model(**inputs).logits
                    else:
                        x = batch[0].to(self.device)
                        outputs = self.model(x)
                    preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
        return np.array(preds)

    def predict(self, test_df) -> Tuple[np.ndarray, np.ndarray]:
        test_texts = test_df['text'].tolist()
        test_labels = test_df['label'].tolist() if 'label' in test_df.columns else [np.nan]*len(test_texts)
        preds = self.predict_from_texts(test_texts)
        return preds, np.array(test_labels)


def run_pipeline(train_df=None, val_df=None, num_classes=None):
    """
    Run the AutoML pipeline with the provided parameters.
    """
    
    # Load train and validation dat

    if train_df is None or val_df is None or num_classes is None:
        raise ValueError("train_df, val_df, and num_classes must be provided.")

    def _objective_function(**kwargs):
        automl = TextAutoML(
            num_classes=num_classes,
            seed=kwargs.get('seed', 42),
            approach=kwargs.get('approach', 'auto'),
            vocab_size=kwargs.get('vocab_size', 10000),
            token_length=kwargs.get('token_length', 128),
            epochs=kwargs.get('epochs', 5),
            batch_size=kwargs.get('batch_size', 64),
            lr=kwargs.get('lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 0.0),
            ffnn_hidden=kwargs.get('ffnn_hidden', 128),
            lstm_emb_dim=kwargs.get('lstm_emb_dim', 128),
            lstm_hidden_dim=kwargs.get('lstm_hidden_dim', 128)
        )
        val_acc = automl.fit(
            train_df=train_df,
            val_df=val_df,
            num_classes=num_classes,
        )
        return 1 - val_acc

    pipeline_space = {
        'approach': neps.Categorical(choices=['tfidf', 'lstm', 'transformer']),
        'vocab_size': neps.Integer(lower=5000, upper=20000, log=False),
        'token_length': neps.Integer(lower=64, upper=256, log=False),
        'epochs': neps.Integer(lower=3, upper=10, log=False),
        'batch_size': neps.Integer(lower=16, upper=64, log=False),
        'lr': neps.Float(lower=1e-4, upper=1e-3, log=False),
        'weight_decay': neps.Float(lower=0.0, upper=1e-5, log=False),
        'ffnn_hidden': neps.Integer(lower=64, upper=256, log=False),
        'lstm_emb_dim': neps.Integer(lower=50, upper=200, log=False),
        'lstm_hidden_dim': neps.Integer(lower=50, upper=200, log=False)
    }

    neps.run(
        evaluate_pipeline=_objective_function,
        pipeline_space=pipeline_space,
        root_directory="./neps/results_text_automl",
        max_evaluations_total=20,
        overwrite_working_directory=True,
        post_run_summary=True,
    )
    
    # return the best parameters found
    try: 
        best_params = neps.get_best_parameters()
        if best_params:
            return best_params
        else:
            logger.error("No best parameters found.")
        return None     