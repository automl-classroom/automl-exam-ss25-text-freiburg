import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from automl.models import SimpleFFNN, LSTMClassifier
from automl.utils import SimpleTextDataset
import logging
from typing import Tuple
from collections import Counter

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TextAutoML:
    def __init__(
        self,
        seed=42,
        approach='auto',
        vocab_size=10000,
        token_length=128,
        epochs=5,
        batch_size=64,
        lr=1e-4,
        weight_decay=0.0,
        ffnn_hidden=128,
        lstm_emb_dim=128,
        lstm_hidden_dim=128,
        fraction_layers_to_finetune: float=1.0,
    ):
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
        self.fraction_layers_to_finetune = fraction_layers_to_finetune

        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self.num_classes = None
        self.val_texts = []
        self.val_labels = []

    def fit(
        self,
        train_df,
        val_df,
        num_classes,
        approach=None,
        vocab_size=None,
        token_length=None,
        epochs=None,
        batch_size=None,
        lr=None,
        weight_decay=None,
        ffnn_hidden=None,
        lstm_emb_dim=None,
        lstm_hidden_dim=None,
        fraction_layers_to_finetune=None
    ):
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
        if fraction_layers_to_finetune is not None: self.fraction_layers_to_finetune = fraction_layers_to_finetune
        
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

        dataset = None
        if self.approach == 'tfidf':
            # TODO: check role of vocab size here/max features
            self.vectorizer = TfidfVectorizer(
                max_features=self.vocab_size,
                lowercase=True,
                min_df=2,    # ignore words appearing in less than 2 sentences
                max_df=0.8,  # ignore words appearing in > 80% of sentences
                sublinear_tf=True,  # use log-spaced term-frequency scoring
            )
            X = self.vectorizer.fit_transform(train_texts).toarray()
            dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(train_labels))
            self.model = SimpleFFNN(X.shape[1], hidden=self.ffnn_hidden, output_dim=self.num_classes)

        elif self.approach in ['lstm', 'transformer']:
            model_name = 'distilbert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.vocab_size = self.tokenizer.vocab_size
            dataset = SimpleTextDataset(train_texts, train_labels, self.tokenizer, self.token_length)
            # TODO: check role of token length here

            match self.approach:
                case "lstm":
                    self.model = LSTMClassifier(len(self.tokenizer), self.lstm_emb_dim, self.lstm_hidden_dim, self.num_classes)
                case "transformer":
                    if TRANSFORMERS_AVAILABLE:
                        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_classes)
                        freeze_layers(self.model, self.fraction_layers_to_finetune)  
                    else:
                        raise ValueError(
                            "Need `AutoTokenizer`, `AutoModelForSequenceClassification` "
                            "from `transformers` package."
                        )
                case _:
                    raise ValueError("Unsupported approach or missing transformers.")
        # elif self.approach == 'transformer' and TRANSFORMERS_AVAILABLE:
        #     # model_name = 'distilbert-base-uncased'
        #     # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        #     # TODO: check role of token length here
        #     dataset = SimpleTextDataset(train_texts, train_labels, self.tokenizer, self.token_length)
        #     self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=self.num_classes)
        #     freeze_layers(self.model, fraction_layers_to_finetune)  

        # else:
        #     raise ValueError("Unsupported approach or missing transformers.")

        # Training and validating
        self.model.to(self.device)
        assert dataset is not None, f"`dataset` cannot be None here!"
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        val_acc = self._train_loop(loader)

        return 1 - val_acc

    def _train_loop(self, loader):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.epochs):
            total_loss = 0
            for batch in loader:
                self.model.train()
                optimizer.zero_grad()
                print(isinstance(batch, dict), type(batch))
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


def freeze_layers(model, fraction_layers_to_finetune: float=1.0) -> None:
    total_layers = len(model.distilbert.transformer.layer)
    _num_layers_to_finetune = int(fraction_layers_to_finetune * total_layers)
    layers_to_freeze = total_layers - _num_layers_to_finetune

    for layer in model.distilbert.transformer.layer[:layers_to_freeze]:
        for param in layer.parameters():
            param.requires_grad = False
# end of file