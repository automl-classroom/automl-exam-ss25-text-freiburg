"""
Download and prepare AG News, IMDB, Amazon, Yelp, and DBpedia datasets for text classification.
Creates train/test splits with fixed seeds for reproducibility.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import requests
import gzip
import tarfile
import zipfile
from sklearn.model_selection import train_test_split
import logging
import argparse
from tqdm import tqdm
import re
import json

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print(
        "Hugging Face datasets not available. "
        "Some datasets will be unavailable. "
        "Install with: pip install datasets."
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    def __init__(self, data_dir: Path = Path("data"), seed: int = 42):
        self.data_dir = data_dir
        self.seed = seed
        self.data_dir.mkdir(exist_ok=True)
        
    def download_file(self, url: str, filepath: Path) -> None:
        """Download a file with progress bar."""
        if filepath.exists():
            logger.info(f"File {filepath} already exists, skipping download.")
            return
            
        logger.info(f"Downloading {url} to {filepath}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def extract_archive(self, archive_path: Path, extract_to: Path) -> None:
        """Extract various archive formats."""
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix == '.gz' and archive_path.stem.endswith('.tar'):
            with tarfile.open(archive_path, 'r:gz') as tar:
                tar.extractall(extract_to)
        elif archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix == '.gz':
            with gzip.open(archive_path, 'rb') as f_in:
                output_path = extract_to / archive_path.stem
                with open(output_path, 'wb') as f_out:
                    f_out.write(f_in.read())
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if pd.isna(text):
            return ""
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', str(text))
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def shuffle_within_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shuffle data within a split using consistent seeding.
        
        Args:
            df: DataFrame to shuffle
            
        Returns:
            Shuffled DataFrame with reset index
        """
        return df.sample(n=len(df), random_state=self.seed).reset_index(drop=True)
    
    def save_train_test_directly(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dataset_name: str,
    ) -> None:
        """Save train and test DataFrames directly without re-splitting."""
        output_dir = self.data_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        train_path = output_dir / "train.csv"
        test_path = output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        with open(output_dir / ".seed.txt", "w") as f:
            f.writelines(f"seed_used: {self.seed}")

        logger.info(f"Saved {dataset_name}:")
        logger.info(f"  Train: {len(train_df)} samples -> {train_path}")
        logger.info(f"  Test: {len(test_df)} samples -> {test_path}")
        logger.info(f"  Classes: {sorted(train_df['label'].unique())}")


class AGNewsDownloader(DatasetDownloader):
    """Download and prepare AG News dataset."""
    
    def download_and_prepare(self):
        logger.info("Preparing AG News dataset...")
        
        # AG News URLs
        # source: https://github.com/mhjabreel/CharCnn_Keras/blob/master/data/ag_news_csv/readme.txt
        # 4 classes
        # 120k train + 7.6k test samples
        urls = {
            'train': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv',
            'test': 'https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv'
        }
        
        # Download files
        temp_dir = self.data_dir / ".cache" / "ag_news"
        train_path = temp_dir / "train_raw.csv"
        test_path = temp_dir / "test_raw.csv"
        
        self.download_file(urls['train'], train_path)
        self.download_file(urls['test'], test_path)
        
        # Load and process
        train_raw = pd.read_csv(train_path, header=None, names=['label', 'title', 'description'])
        test_raw = pd.read_csv(test_path, header=None, names=['label', 'title', 'description'])
        
        # Process AG News data
        def process_ag_news(df):
            df['text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
            df['text'] = df['text'].apply(self.clean_text)
            df['label'] = df['label'] - 1  # Convert from 1-4 to 0-3
            return df[['text', 'label']].copy()
        
        train_df = process_ag_news(train_raw)
        test_df = process_ag_news(test_raw)
        
        # Remove empty texts
        train_df = train_df[train_df['text'].str.strip() != '']
        test_df = test_df[test_df['text'].str.strip() != '']
        
        logger.info(f"AG News - Train samples: {len(train_df)}")
        logger.info(f"AG News - Test samples: {len(test_df)}")
        logger.info(f"AG News - Total samples: {len(train_df) + len(test_df)}")
        logger.info(f"AG News - Class distribution:\n{train_df['label'].value_counts().sort_index()}")
        
        train_df = self.shuffle_within_split(train_df)
        test_df = self.shuffle_within_split(test_df)
        self.save_train_test_directly(train_df, test_df, 'ag_news')


class IMDBDownloader(DatasetDownloader):
    """Download and prepare IMDB dataset."""
    
    def download_and_prepare(self):
        logger.info("Preparing IMDB dataset...")
        
        # IMDB dataset URL
        # source: https://huggingface.co/datasets/stanfordnlp/imdb#dataset-summary
        # 2 classes
        # 25k train + 25k test samples
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        archive_path = self.data_dir / ".cache" / "aclImdb_v1.tar.gz"
        extract_dir = self.data_dir / ".cache" / "imdb_extracted"
        
        # Download and extract
        self.download_file(url, archive_path)
        self.extract_archive(archive_path, extract_dir)
        
        # Process IMDB data
        imdb_dir = extract_dir / "aclImdb"
        
        def load_imdb_split(split_dir, sentiment):
            """Load positive or negative reviews from a directory."""
            texts = []
            sentiment_dir = split_dir / sentiment
            
            if not sentiment_dir.exists():
                return [], []
            
            for file_path in sentiment_dir.glob("*.txt"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        texts.append(self.clean_text(text))
                except Exception as e:
                    logger.warning(f"Error reading {file_path}: {e}")
                    continue
            
            labels = [1 if sentiment == 'pos' else 0] * len(texts)
            return texts, labels
        
        # Load all data
        all_texts = {}
        all_labels = {}

        for split in ['train', 'test']:
            all_texts[split] = []
            all_labels[split] = []
            split_dir = imdb_dir / split
            if split_dir.exists():
                # Load positive reviews
                pos_texts, pos_labels = load_imdb_split(split_dir, 'pos')
                all_texts[split].extend(pos_texts)
                all_labels[split].extend(pos_labels)
                
                # Load negative reviews
                neg_texts, neg_labels = load_imdb_split(split_dir, 'neg')
                all_texts[split].extend(neg_texts)
                all_labels[split].extend(neg_labels)
        
        # Create DataFrames
        train_df = pd.DataFrame({
            'text': all_texts["train"],
            'label': all_labels["train"]
        })
        test_df = pd.DataFrame({
            'text': all_texts["test"],
            'label': all_labels["test"]
        })
        
        # Remove empty texts
        train_df = train_df[train_df['text'].str.strip() != '']
        test_df = test_df[test_df['text'].str.strip() != '']
        
        logger.info(f"IMDB - Train samples: {len(train_df)}")
        logger.info(f"IMDB - Test samples: {len(test_df)}")
        logger.info(f"IMDB - Total samples: {len(train_df) + len(test_df)}")
        logger.info(f"IMDB - Class distribution:\n{train_df['label'].value_counts().sort_index()}")
        
        train_df = self.shuffle_within_split(train_df)
        test_df = self.shuffle_within_split(test_df)
        self.save_train_test_directly(train_df, test_df, 'imdb')


class AmazonDownloader(DatasetDownloader):
    """Download and prepare Amazon reviews dataset."""
    
    def download_and_prepare(self):
        logger.info("Preparing Amazon reviews dataset...")
        
        # Use Amazon product reviews for category classification
        # This uses a subset of Julian McAuley's Amazon review dataset
        # source: https://snap.stanford.edu/data/web-Amazon.html
        # 3 classes (converted from 1-5 star ratings)
        url = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Amazon_Instant_Video_5.json.gz"
        
        self._download_real_amazon(url)

    def _download_real_amazon(self, url):
        """Download real Amazon dataset."""
        archive_path = self.data_dir / ".cache" / "amazon_reviews.json.gz"
        
        # Download
        self.download_file(url, archive_path)
        
        # Parse JSON lines
        reviews = []
        with gzip.open(archive_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    if 'reviewText' in review and 'overall' in review:
                        reviews.append({
                            'text': self.clean_text(review['reviewText']),
                            'rating': review['overall']
                        })
                except json.JSONDecodeError:
                    continue
        
        # Convert ratings to categories (1-2: negative, 3: neutral, 4-5: positive)
        df = pd.DataFrame(reviews)
        df = df[df['text'].str.strip() != '']
        
        # Create 3 categories based on ratings
        def rating_to_category(rating):
            if rating <= 2:
                return 0  # Negative
            elif rating == 3:
                return 1  # Neutral
            else:
                return 2  # Positive
        
        df['label'] = df['rating'].apply(rating_to_category)
        df = df[['text', 'label']].copy()
        
        # Create train/test split ONCE (since Amazon doesn't have original splits)
        train_df, test_df = train_test_split(
            df, test_size=0.327, random_state=self.seed, stratify=df['label']
        )
        
        logger.info(f"Amazon - Train samples: {len(train_df)}")
        logger.info(f"Amazon - Test samples: {len(test_df)}")
        logger.info(f"Amazon - Total samples: {len(train_df) + len(test_df)}")
        logger.info(f"Amazon - Class distribution:\n{train_df['label'].value_counts().sort_index()}")
        
        train_df = self.shuffle_within_split(train_df)
        test_df = self.shuffle_within_split(test_df)
        self.save_train_test_directly(train_df, test_df, 'amazon')


class YelpDownloader(DatasetDownloader):
    """Download and prepare Yelp Reviews dataset."""
    
    def download_and_prepare(self):
        logger.info("Preparing Yelp Reviews (5-star) dataset...")
        
        # Yelp Reviews dataset
        # source: https://huggingface.co/datasets/yelp_review_full
        # 5 classes (1-5 star ratings, 0-indexed as 0-4)
        # 650k train + 50k test samples originally
        
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets required for Yelp dataset")
            return
            
        self._download_real_yelp()
    
    def _download_real_yelp(self):
        """Download real Yelp dataset using Hugging Face."""
        logger.info("Downloading Yelp dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("yelp_review_full", cache_dir=self.data_dir / ".cache")
        
        # Convert to pandas
        train_data = dataset['train'].to_pandas()
        test_data = dataset['test'].to_pandas()
        
        # Clean up data
        train_data = train_data.rename(columns={'text': 'text', 'label': 'label'})
        test_data = test_data.rename(columns={'text': 'text', 'label': 'label'})
        train_data['text'] = train_data['text'].astype(str).apply(self.clean_text)
        test_data['text'] = test_data['text'].astype(str).apply(self.clean_text)
        train_data['label'] = train_data['label'].astype(int)
        test_data['label'] = test_data['label'].astype(int)
        
        # Remove empty texts
        train_df = train_data[train_data['text'].str.strip() != '']
        test_df = test_data[test_data['text'].str.strip() != '']
        
        logger.info(f"Yelp - Train samples: {len(train_df)}")
        logger.info(f"Yelp - Test samples: {len(test_df)}")
        logger.info(f"Yelp - Total samples: {len(train_df) + len(test_df)}")
        logger.info(f"Yelp - Class distribution:\n{train_df['label'].value_counts().sort_index()}")
        
        train_df = self.shuffle_within_split(train_df)
        test_df = self.shuffle_within_split(test_df)
        self.save_train_test_directly(train_df, test_df, 'yelp')


class DBpediaDownloader(DatasetDownloader):
    """Download and prepare DBpedia ontology classification dataset."""
    
    def download_and_prepare(self):
        logger.info("Preparing DBpedia ontology classification dataset...")
        
        # DBpedia ontology dataset
        # source: https://huggingface.co/datasets/fancyzhx/dbpedia_14
        # 14 classes (Company, EducationalInstitution, Artist, Athlete, OfficeHolder, 
        #             MeanOfTransportation, Building, NaturalPlace, Village, Animal, 
        #             Plant, Album, Film, WrittenWork)
        # 560k train + 70k test samples originally (40k train + 5k test per class)
        
        if not HF_AVAILABLE:
            logger.error("Hugging Face datasets required for DBpedia dataset")
            return
            
        self._download_real_dbpedia()
    
    def _download_real_dbpedia(self):
        """Download real DBpedia dataset using Hugging Face."""
        logger.info("Downloading DBpedia dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("fancyzhx/dbpedia_14", cache_dir=self.data_dir / ".cache")
        
        # Convert to pandas
        train_data = dataset['train'].to_pandas()
        test_data = dataset['test'].to_pandas()
        
        # Combine title and content for text field
        def combine_text(row):
            title = str(row.get('title', ''))
            content = str(row.get('content', ''))
            return f"{title} {content}".strip()
        
        train_data['text'] = train_data.apply(combine_text, axis=1)
        test_data['text'] = test_data.apply(combine_text, axis=1)
        
        # Clean up data
        train_data['text'] = train_data['text'].astype(str).apply(self.clean_text)
        test_data['text'] = test_data['text'].astype(str).apply(self.clean_text)
        train_data['label'] = train_data['label'].astype(int) - 1  # Convert to 0-based indexing
        test_data['label'] = test_data['label'].astype(int) - 1
        
        # Select only needed columns
        train_df = train_data[['text', 'label']].copy()
        test_df = test_data[['text', 'label']].copy()
        
        # Remove empty texts
        train_df = train_df[train_df['text'].str.strip() != '']
        test_df = test_df[test_df['text'].str.strip() != '']
        
        logger.info(f"DBpedia - Train samples: {len(train_df)}")
        logger.info(f"DBpedia - Test samples: {len(test_df)}")
        logger.info(f"DBpedia - Total samples: {len(train_df) + len(test_df)}")
        logger.info(f"DBpedia - Class distribution:\n{train_df['label'].value_counts().sort_index()}")
        
        train_df = self.shuffle_within_split(train_df)
        test_df = self.shuffle_within_split(test_df)
        self.save_train_test_directly(train_df, test_df, 'dbpedia')
    

def main():
    parser = argparse.ArgumentParser(description="Download and prepare text datasets")
    parser.add_argument(
        "--datasets", 
        nargs='+', 
        default=['ag_news', 'imdb', 'amazon', 'yelp', 'dbpedia'],
        choices=['ag_news', 'imdb', 'amazon', 'yelp', 'dbpedia'],
        help="Datasets to download and prepare"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to save datasets"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    
    args = parser.parse_args()
    
    # Track success/failure
    results = {}
    
    # Create downloaders
    if 'ag_news' in args.datasets:
        logger.info("\n" + "="*50)
        logger.info("DOWNLOADING AG NEWS DATASET")
        logger.info("="*50)
        try:
            downloader = AGNewsDownloader(args.data_dir, args.seed)
            downloader.download_and_prepare()
            results['ag_news'] = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to download AG News: {e}")
            results['ag_news'] = 'FAILED'
    
    if 'imdb' in args.datasets:
        logger.info("\n" + "="*50)
        logger.info("DOWNLOADING IMDB DATASET")
        logger.info("="*50)
        try:
            downloader = IMDBDownloader(args.data_dir, args.seed)
            downloader.download_and_prepare()
            results['imdb'] = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to download IMDB: {e}")
            results['imdb'] = 'FAILED'
    
    if 'amazon' in args.datasets:
        logger.info("\n" + "="*50)
        logger.info("DOWNLOADING AMAZON DATASET")
        logger.info("="*50)
        try:
            downloader = AmazonDownloader(args.data_dir, args.seed)
            downloader.download_and_prepare()
            results['amazon'] = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to download Amazon: {e}")
            results['amazon'] = 'FAILED'

    if 'yelp' in args.datasets:
        logger.info("\n" + "="*50)
        logger.info("DOWNLOADING YELP DATASET")
        logger.info("="*50)
        try:
            downloader = YelpDownloader(args.data_dir, args.seed)
            downloader.download_and_prepare()
            results['yelp'] = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to download Yelp: {e}")
            results['yelp'] = 'FAILED'
    
    if 'dbpedia' in args.datasets:
        logger.info("\n" + "="*50)
        logger.info("DOWNLOADING DBPEDIA DATASET")
        logger.info("="*50)
        try:
            downloader = DBpediaDownloader(args.data_dir, args.seed)
            downloader.download_and_prepare()
            results['dbpedia'] = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to download DBpedia: {e}")
            results['dbpedia'] = 'FAILED'
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("DATASET DOWNLOAD SUMMARY")
    logger.info("="*60)
    
    successful = [name for name, status in results.items() if status == 'SUCCESS']
    failed = [name for name, status in results.items() if status == 'FAILED']
    
    logger.info(f"✅ Successfully downloaded: {len(successful)}/{len(results)} datasets")
    for dataset in successful:
        logger.info(f"   ✓ {dataset}")
    
    if failed:
        logger.info(f"❌ Failed downloads: {len(failed)} datasets")
        for dataset in failed:
            logger.info(f"   ✗ {dataset}")
    
    logger.info(f"\n📁 All data saved to: {args.data_dir}")
    logger.info("\n🚀 You can now run your AutoML experiments:")
    for dataset in successful:
        logger.info(f"   python run_text.py --dataset {dataset}")
    
    if HF_AVAILABLE:
        logger.info(f"\n💡 Tip: Hugging Face datasets available - real data downloaded when possible")
    else:
        logger.info(f"\n⚠️  Install 'datasets' package for real Yelp and DBpedia data: pip install datasets")


if __name__ == "__main__":
    main()
