"""
Verify that datasets have been downloaded and prepared correctly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_dataset(data_dir: Path, dataset_name: str, expected_classes: int) -> bool:
    """Verify a single dataset."""
    dataset_dir = data_dir / dataset_name
    train_path = dataset_dir / "train.csv"
    test_path = dataset_dir / "test.csv"
    
    logger.info(f"\n=== Verifying {dataset_name.upper()} ===")
    
    # Check if files exist
    if not train_path.exists():
        logger.error(f"Train file not found: {train_path}")
        return False
    
    if not test_path.exists():
        logger.error(f"Test file not found: {test_path}")
        return False
    
    try:
        # Load datasets
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Check required columns
        required_cols = ['text', 'label']
        for df_name, df in [('train', train_df), ('test', test_df)]:
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                logger.error(f"{df_name} missing columns: {missing_cols}")
                return False
        
        # Check data integrity
        for df_name, df in [('train', train_df), ('test', test_df)]:
            # Check for missing values
            if df['text'].isna().any():
                logger.warning(f"{df_name} has missing text values")
            
            if df['label'].isna().any():
                logger.error(f"{df_name} has missing label values")
                return False
            
            # Check label range
            unique_labels = sorted(df['label'].unique())
            expected_labels = list(range(expected_classes))
            
            if unique_labels != expected_labels:
                logger.error(f"{df_name} labels {unique_labels} != expected {expected_labels}")
                return False
            
            # Check for empty texts
            empty_texts = (df['text'].str.strip() == '').sum()
            if empty_texts > 0:
                logger.warning(f"{df_name} has {empty_texts} empty texts")
        
        # Print statistics
        logger.info(f"✓ Files exist and are readable")
        logger.info(f"✓ Required columns present: {required_cols}")
        logger.info(f"✓ Labels are correct: {expected_labels}")
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        logger.info(f"Total samples: {len(train_df) + len(test_df)}")
        
        # Class distribution
        logger.info("Train class distribution:")
        for label in sorted(train_df['label'].unique()):
            count = (train_df['label'] == label).sum()
            pct = count / len(train_df) * 100
            logger.info(f"  Class {label}: {count} ({pct:.1f}%)")
        
        logger.info("Test class distribution:")
        for label in sorted(test_df['label'].unique()):
            count = (test_df['label'] == label).sum()
            pct = count / len(test_df) * 100
            logger.info(f"  Class {label}: {count} ({pct:.1f}%)")
        
        # Text length statistics
        train_lengths = train_df['text'].str.len()
        test_lengths = test_df['text'].str.len()
        
        logger.info(f"Text length stats (train): "
                   f"min={train_lengths.min()}, "
                   f"max={train_lengths.max()}, "
                   f"mean={train_lengths.mean():.1f}")
        
        logger.info(f"Text length stats (test): "
                   f"min={test_lengths.min()}, "
                   f"max={test_lengths.max()}, "
                   f"mean={test_lengths.mean():.1f}")
        
        # Sample texts
        logger.info("Sample train texts:")
        for i, (text, label) in enumerate(zip(train_df['text'].head(3), train_df['label'].head(3))):
            text_preview = text[:100] + "..." if len(text) > 100 else text
            logger.info(f"  [{label}] {text_preview}")
        
        logger.info(f"✓ {dataset_name.upper()} verification passed!")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying {dataset_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Verify downloaded datasets")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs='+',
        default=['ag_news', 'imdb', 'amazon'],
        choices=['ag_news', 'imdb', 'amazon'],
        help="Datasets to verify"
    )
    
    args = parser.parse_args()
    
    # Dataset specifications
    dataset_specs = {
        'ag_news': {'classes': 4, 'name': 'AG News'},
        'imdb': {'classes': 2, 'name': 'IMDB'},
        'amazon': {'classes': 5, 'name': 'Amazon Reviews'}  # Adjust based on your Amazon setup
    }
    
    all_passed = True
    
    for dataset_name in args.datasets:
        if dataset_name in dataset_specs:
            specs = dataset_specs[dataset_name]
            passed = verify_dataset(args.data_dir, dataset_name, specs['classes'])
            all_passed = all_passed and passed
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All dataset verifications passed!")
        logger.info("You can now run your AutoML experiments.")
    else:
        logger.error("\n❌ Some dataset verifications failed!")
        logger.error("Please check the error messages above and re-download if necessary.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())