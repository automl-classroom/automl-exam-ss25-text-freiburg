"""
Simple setup script that downloads text classification datasets.
"""

import argparse
from pathlib import Path
from download_datasets import AGNewsDownloader, IMDBDownloader, AmazonDownloader, YelpDownloader, DBpediaDownloader


def main():
    parser = argparse.ArgumentParser(description="Download text datasets")
    parser.add_argument(
        "--dataset", 
        nargs='+', 
        choices=['ag_news', 'imdb', 'amazon', 'yelp', 'dbpedia', 'all'],
        default=['all'],
        help="Datasets to download (default: all)"
    )
    parser.add_argument("--data_dir", type=Path, default=Path(".data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Expand 'all' to all datasets
    if 'all' in args.dataset:
        datasets = ['ag_news', 'imdb', 'amazon', 'yelp', 'dbpedia']
    else:
        datasets = args.dataset
    
    downloaders = {
        'ag_news': AGNewsDownloader,
        'imdb': IMDBDownloader,
        'amazon': AmazonDownloader,  # omit this
        'yelp': YelpDownloader,  # potential test set
        'dbpedia': DBpediaDownloader
    }
    
    print(f"Downloading datasets: {', '.join(datasets)}")
    
    for dataset in datasets:
        print(f"Downloading {dataset}...")
        downloaders[dataset](args.data_dir, args.seed).download_and_prepare()
    
    print("Done! Run: python run_text.py --dataset <dataset_name>")


if __name__ == "__main__":
    main()