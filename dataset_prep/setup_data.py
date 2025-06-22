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
        default=None,
        help="Datasets to download (default: None [throws error])"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=None,
        help="Location to download (default: `cwd`/.data/"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    downloaders = {
        'ag_news': AGNewsDownloader,
        'imdb': IMDBDownloader,
        'amazon': AmazonDownloader,  # omit this
        'yelp': YelpDownloader,  # potential test set
        'dbpedia': DBpediaDownloader
    }
    
    assert args.dataset is not None, f"Specify `all` explicitly or list of dataset(s) (see --help)."
    print(f"Downloading datasets: {', '.join(args.dataset)}")
    
    if len(args.dataset) == 1 and args.dataset[0].lower() == "all":
        args.dataset = list(downloaders.keys())

    if args.data_dir is None:
        args.data_dir = Path.cwd().absolute() / ".data"

    for dataset in args.dataset:
        print(f"Downloading {dataset} to {args.data_dir}")
        downloaders[dataset](args.data_dir, args.seed).download_and_prepare()
    
    print("Done!")


if __name__ == "__main__":
    main()