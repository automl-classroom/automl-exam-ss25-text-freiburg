"""
Test the complete AutoML pipeline end-to-end.
"""

import subprocess
import sys
from pathlib import Path
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd, description):
    """Run a command and check its success."""
    logger.info(f"\n=== {description} ===")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("✓ Command completed successfully")
        if result.stdout:
            logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Command failed with return code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False


def test_data_setup():
    """Test data download and verification."""
    logger.info("Testing data setup...")
    
    # Test setup script
    if not run_command([sys.executable, "setup_data.py"], "Data Setup"):
        return False
    
    # Test verification
    if not run_command([sys.executable, "verify_data.py"], "Data Verification"):
        return False
    
    return True


def test_automl_runs():
    """Test AutoML runs on all datasets."""
    logger.info("Testing AutoML runs...")
    
    datasets = ["ag_news", "imdb", "amazon"]
    
    for dataset in datasets:
        cmd = [
            sys.executable, "run_text.py",
            "--dataset", dataset,
            "--output-path", f"test_{dataset}_predictions.npy",
            "--seed", "42"
        ]
        
        if not run_command(cmd, f"AutoML on {dataset}"):
            return False
        
        # Check if prediction file was created
        pred_file = Path(f"test_{dataset}_predictions.npy")
        if pred_file.exists():
            logger.info(f"✓ Prediction file created: {pred_file}")
        else:
            logger.error(f"❌ Prediction file not created: {pred_file}")
            return False
    
    return True


def cleanup_test_files():
    """Clean up test files."""
    logger.info("Cleaning up test files...")
    
    test_files = [
        "test_ag_news_predictions.npy",
        "test_imdb_predictions.npy", 
        "test_amazon_predictions.npy"
    ]
    
    for file_path in test_files:
        path = Path(file_path)
        if path.exists():
            path.unlink()
            logger.info(f"Removed: {file_path}")


def main():
    parser = argparse.ArgumentParser(description="Test the complete AutoML pipeline")
    parser.add_argument(
        "--skip-data-setup",
        action="store_true",
        help="Skip data download and setup (assumes data already exists)"
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up test files after completion"
    )
    
    args = parser.parse_args()
    
    logger.info("🧪 Starting AutoML Pipeline Test")
    
    success = True
    
    # Test data setup
    if not args.skip_data_setup:
        logger.info("\n1. Testing data setup...")
        if not test_data_setup():
            logger.error("Data setup test failed")
            success = False
    else:
        logger.info("Skipping data setup (--skip-data-setup specified)")
    
    # Test AutoML runs
    if success:
        logger.info("\n2. Testing AutoML runs...")
        if not test_automl_runs():
            logger.error("AutoML runs test failed")
            success = False
    
    # Cleanup
    if args.cleanup:
        cleanup_test_files()
    
    # Summary
    if success:
        logger.info("\n🎉 All tests passed! Your AutoML pipeline is working correctly.")
        logger.info("\nYou can now:")
        logger.info("  • Run experiments: python run_text.py --dataset ag_news")
        logger.info("  • Modify the AutoML system in src/automl.py")
        logger.info("  • Add new datasets in src/datasets.py")
        logger.info("  • Experiment with different approaches and hyperparameters")
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())