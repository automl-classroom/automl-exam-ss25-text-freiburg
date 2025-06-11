"""
Simple setup script that downloads datasets with automatic fallbacks.
Tries Hugging Face first, then falls back to direct downloads.
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False


def check_and_install_requirements():
    """Check and install required packages."""
    required_packages = [
        "torch", "scikit-learn", "numpy", "pandas", 
        "requests", "tqdm"
    ]
    
    optional_packages = ["datasets", "transformers"]
    
    logger.info("Checking required packages...")
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package} is available")
        except ImportError:
            logger.info(f"Installing {package}...")
            if install_package(package):
                logger.info(f"✓ {package} installed successfully")
            else:
                logger.error(f"❌ Failed to install {package}")
                return False
    
    logger.info("Checking optional packages...")
    
    for package in optional_packages:
        try:
            __import__(package.replace("-", "_"))
            logger.info(f"✓ {package} is available")
        except ImportError:
            logger.info(f"Installing optional package {package}...")
            if install_package(package):
                logger.info(f"✓ {package} installed successfully")
            else:
                logger.warning(f"⚠️ Optional package {package} could not be installed")
    
    return True


def download_datasets(data_dir="data", seed=42):
    """Download datasets using the best available method."""
    logger.info("Starting dataset download...")
    
    # Try Hugging Face first
    try:
        from download_with_huggingface import HuggingFaceDownloader
        
        logger.info("Using Hugging Face datasets...")
        downloader = HuggingFaceDownloader(Path(data_dir), seed)
        
        try:
            downloader.download_ag_news()
            logger.info("✓ AG News downloaded successfully")
        except Exception as e:
            logger.error(f"❌ AG News download failed: {e}")
        
        try:
            downloader.download_imdb()
            logger.info("✓ IMDB downloaded successfully")
        except Exception as e:
            logger.error(f"❌ IMDB download failed: {e}")
        
        try:
            downloader.download_amazon()
            logger.info("✓ Amazon reviews downloaded successfully")
        except Exception as e:
            logger.error(f"❌ Amazon reviews download failed: {e}")
            
    except ImportError:
        logger.warning("Hugging Face datasets not available, trying direct downloads...")
        
        # Fallback to direct downloads
        try:
            from download_datasets import AGNewsDownloader, IMDBDownloader, AmazonDownloader
            
            logger.info("Using direct download method...")
            
            try:
                downloader = AGNewsDownloader(Path(data_dir), seed)
                downloader.download_and_prepare()
                logger.info("✓ AG News downloaded successfully")
            except Exception as e:
                logger.error(f"❌ AG News download failed: {e}")
            
            try:
                downloader = IMDBDownloader(Path(data_dir), seed)
                downloader.download_and_prepare()
                logger.info("✓ IMDB downloaded successfully")
            except Exception as e:
                logger.error(f"❌ IMDB download failed: {e}")
            
            try:
                downloader = AmazonDownloader(Path(data_dir), seed)
                downloader.download_and_prepare()
                logger.info("✓ Amazon reviews downloaded successfully")
            except Exception as e:
                logger.error(f"❌ Amazon reviews download failed: {e}")
                
        except ImportError as e:
            logger.error(f"Could not import download modules: {e}")
            return False
    
    return True


def verify_downloads(data_dir="data"):
    """Verify that downloads completed successfully."""
    logger.info("Verifying downloads...")
    
    try:
        from verify_data import verify_dataset
        
        dataset_specs = {
            'ag_news': 4,
            'imdb': 2,
            'amazon': 5
        }
        
        all_passed = True
        for dataset_name, num_classes in dataset_specs.items():
            passed = verify_dataset(Path(data_dir), dataset_name, num_classes)
            all_passed = all_passed and passed
        
        return all_passed
        
    except ImportError as e:
        logger.error(f"Could not import verification module: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("=== Text AutoML Dataset Setup ===")
    
    # Step 1: Check and install requirements
    logger.info("\n1. Checking requirements...")
    if not check_and_install_requirements():
        logger.error("Failed to install required packages")
        return 1
    
    # Step 2: Download datasets
    logger.info("\n2. Downloading datasets...")
    if not download_datasets():
        logger.error("Dataset download failed")
        return 1
    
    # Step 3: Verify downloads
    logger.info("\n3. Verifying downloads...")
    if not verify_downloads():
        logger.error("Dataset verification failed")
        return 1
    
    logger.info("\n🎉 Setup completed successfully!")
    logger.info("You can now run your AutoML experiments with:")
    logger.info("  python run_text.py --dataset ag_news")
    logger.info("  python run_text.py --dataset imdb")
    logger.info("  python run_text.py --dataset amazon")
    
    return 0


if __name__ == "__main__":
    exit(main())