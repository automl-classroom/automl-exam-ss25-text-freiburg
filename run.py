"""An example run file which trains a dummy NLP AutoML system on the training split of a dataset
and logs the accuracy score on the test set.

In the example data you are given access to the labels of the test split, however
in the test dataset we will provide later, you will not have access
to this and you will need to output your predictions for the text of the test set
to a file, which we will grade using github classrooms!
"""
from __future__ import annotations

from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import argparse
import logging

from src.automl import TextAutoML, run_pipeline
from src.datasets import (
    AGNewsDataset,
    AmazonReviewsDataset,
    DBpediaDataset,
    IMDBDataset,
    YelpDataset
)

logger = logging.getLogger(__name__)


def main(
        dataset: str,
        output_path: Path,
        seed: int,
        approach: str,
        vocab_size: int = 10000,
        token_length: int = 128,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.0001,
        weight_decay: float = 0.01,
        ffnn_hidden: int = 128,
        lstm_emb_dim: int = 128,
        lstm_hidden_dim: int = 128,
        hpo: bool = False
    ):
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case "dbpedia":
            dataset_class = DBpediaDataset
        case "yelp":
            dataset_class = YelpDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Fitting Text AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.

    # Get the dataset and create dataloaders
    data_info = dataset_class().create_dataloaders()
    train_df = data_info['train_df']
    val_df = data_info.get('val_df', None)
    test_df = data_info['test_df']
    num_classes = data_info['num_classes']
    logger.info(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    logger.info(f"Number of classes: {num_classes}")

    if hpo:
        #NEPS PART
        logger.info("Running hyperparameter optimization with NePS")
        best_params = run_pipeline(
            train_df=train_df,
            val_df=val_df,
            num_classes=num_classes, 
        )
        logger.info(f"Best parameters from NePS: {best_params}")

        approach = best_params.get('approach', approach)
        seed = best_params.get('seed', seed)
        vocab_size = best_params.get('vocab_size', vocab_size)
        token_length = best_params.get('token_length', token_length)
        epochs = best_params.get('epochs', epochs)
        batch_size = best_params.get('batch_size', batch_size)
        lr = best_params.get('lr', lr)
        weight_decay = best_params.get('weight_decay', weight_decay)
        ffnn_hidden = best_params.get('ffnn_hidden', ffnn_hidden)
        lstm_emb_dim = best_params.get('lstm_emb_dim', lstm_emb_dim)
        lstm_hidden_dim = best_params.get('lstm_hidden_dim', lstm_hidden_dim)

    # Initialize the TextAutoML instance with the best parameters
    automl = TextAutoML(
        seed=seed, 
        approach=approach, 
        vocab_size=vocab_size, 
        token_length=token_length,
        epochs=epochs, 
        batch_size=batch_size,
        lr=lr, 
        weight_decay=weight_decay,
        ffnn_hidden=ffnn_hidden, 
        lstm_emb_dim=lstm_emb_dim,
        lstm_hidden_dim=lstm_hidden_dim
    )

    # Fit the AutoML model on the training and validation datasets
    automl.fit(train_df, val_df, num_classes=num_classes)
    logger.info("Training complete")
    
    # Predict on the test set
    test_preds, test_labels = automl.predict(test_df)

    # Write the predictions of X_test to disk
    logger.info("Writing predictions to disk")
    with output_path.open("wb") as f:
        np.save(f, test_preds)

    # In case of running on the final exam data, also add the predictions.npy
    # to the correct location for auto evaluation.
    if dataset == "amazon":  # Assuming amazon is the exam dataset
        test_output_path = Path("data/exam_text_dataset/predictions.npy")
        test_output_path.parent.mkdir(parents=True, exist_ok=True)
        with test_output_path.open("wb") as f:
            np.save(f, test_preds)

    # Check if test_labels has missing data
    if not np.isnan(test_labels).any():
        acc = accuracy_score(test_labels, test_preds)
        logger.info(f"Accuracy on test set: {acc}")
        
        # Log detailed classification report for better insight
        logger.info("Classification Report:")
        logger.info(f"\n{classification_report(test_labels, test_preds)}")
    else:
        # This is the setting for the exam dataset, you will not have access to the labels
        logger.info(f"No test labels available for dataset '{dataset}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="The name of the dataset to run on.",
        choices=["ag_news", "imdb", "amazon", "dbpedia", "yelp"]
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("text_predictions.npy"),
        help=(
            "The path to save the predictions to."
            " By default this will just save to './text_predictions.npy'."
        )
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed for reproducibility if you are using any randomness,"
            " i.e. torch, numpy, pandas, sklearn, etc."
        )
    )
    parser.add_argument(
        "--approach",
        type=str,
        default="transformer",
        choices=["tfidf", "lstm", "transformer"],
        help=(
            "The approach to use for the AutoML system. "
            "Options are 'tfidf', 'lstm', or 'transformer'."
        )
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10000,
        help="The size of the vocabulary to use for the text dataset."
    )
    parser.add_argument(
        "--token-length",
        type=int,
        default=128,
        help="The maximum length of tokens to use for the text dataset."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="The number of epochs to train the model for."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="The batch size to use for training and evaluation."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="The learning rate to use for the optimizer."
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="The weight decay to use for the optimizer."
    )

    parser.add_argument(
        "--lstm-emb-dim",
        type=int,
        default=128,
        help="The embedding dimension to use for the LSTM model."
    )

    parser.add_argument(
        "--lstm-hidden-dim",
        type=int,
        default=128,
        help="The hidden size to use for the LSTM model."
    )

    parser.add_argument(
        "--ffnn-hidden-layer-dim",
        type=int,
        default=128,
        help="The hidden size to use for the model."
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        default=False,
        help=(
            "Whether to run hyperparameter optimization (HPO) using NePS."
            " If set, the script will use NePS to find the best parameters."
        )
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Whether to log only warnings and errors."
    )
    args = parser.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)

    logger.info(
        f"Running text dataset {args.dataset}"
        f"\n{args}"
    )

    main(
        dataset=args.dataset,
        output_path=args.output_path,
        seed=args.seed,
        approach=args.approach,
        vocab_size=args.vocab_size,
        token_length=args.token_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ffnn_hidden=args.ffnn_hidden_layer_dim,
        lstm_emb_dim=args.lstm_emb_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        hpo=args.hpo
    )