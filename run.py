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

from src import TextAutoML, AGNewsDataset, IMDBDataset, AmazonReviewsDataset

logger = logging.getLogger(__name__)


def main(
        dataset: str,
        output_path: Path,
        seed: int,
):
    match dataset:
        case "ag_news":
            dataset_class = AGNewsDataset
        case "imdb":
            dataset_class = IMDBDataset
        case "amazon":
            dataset_class = AmazonReviewsDataset
        case _:
            raise ValueError(f"Invalid dataset: {dataset}")

    logger.info("Fitting Text AutoML")

    # You do not need to follow this setup or API it's merely here to provide
    # an example of how your automl system could be used.
    # As a general rule of thumb, you should **never** pass in any
    # test data to your AutoML solution other than to generate predictions.
    automl = TextAutoML(seed=seed)
    
    # Load the dataset and create a loader then pass it
    automl.fit(dataset_class)
    
    # Do the same for the test dataset
    test_preds, test_labels = automl.predict(dataset_class)

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
        choices=["ag_news", "imdb", "amazon"]
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
    )