from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import yaml

from sklearn.metrics import accuracy_score, classification_report


FINAL_EXAM_DATASET = "yelp"


def get_text_metadata_from_csv(train_path, text_col="text", label_col="label"):
    df_train = pd.read_csv(train_path)
    
    num_samples = len(df_train)
    labels = df_train[label_col]
    texts = df_train[text_col].astype(str)
    
    num_classes = len(labels.unique())
    sequence_lengths = texts.apply(lambda x: len(x))
    
    min_len = sequence_lengths.min()
    max_len = sequence_lengths.max()

    mean_len = sequence_lengths.mean()
    median_len = sequence_lengths.median()
    class_dist = Counter(labels)
    return num_classes, num_samples, class_dist,  min_len, max_len, mean_len, median_len

def main():
    datasets = [
        {"name": "amazon", "train_path": "./amazon/train.csv"},
        {"name": "imdb", "train_path": "./imdb/train.csv"},
        {"name": "ag_news", "train_path": "./ag_news/train.csv"},
        {"name": "dbpedia", "train_path": "./dbpedia/train.csv"},
        {"name": "yelp", "train_path": "./yelp/train.csv"},
    ]

    print("| Dataset | Labels | Rows | Seq. Length: `min` | Seq. Length: `max` | Seq. Length: `mean` | Seq. Length: `median` |")
    print("| --- | --- |  --- |  --- |  --- | --- | --- |")
    for d in datasets:
        if os.path.exists(d["train_path"]):
            labels, rows, class_dist, min_len, max_len, mean_len, median_len = get_text_metadata_from_csv(d["train_path"])
            print(f"| {d['name']} | {labels} | {rows} | {min_len} | {max_len} | {int(mean_len)} | {int(median_len)} |")  # | {class_dist}")
        else:
            print(f"| {d['name']} | File not found | - | - | - | - | - |")
        
    print("\n" + "=" * 60)
    print(f"Number of samples: {rows}")
    print(f"Number of classes: {labels}")
    print(f"Class distribution: {class_dist}")
    print(f"Sequence length - Min: {min_len}, Max: {max_len}, Mean: {mean_len:.2f}, Median: {median_len}")
    print("=" * 60)


def quick_plot(
    summary_path: Path,
    reference: float=50,
    filename: str=None,
    dataset: str=None,
    ylims: list[float]=None,
) -> None:
    summary_path = Path(summary_path) if isinstance(summary_path, str) else summary_path
    try:
        df = pd.read_csv(summary_path.absolute() / "full.csv")
    except FileNotFoundError:
        import neps
        full, _ = neps.status(summary_path / "..")
        df = full
    except Exception as e:
        raise e
    _prev_len = len(df)
    df = df.dropna()
    print(f"Retaining {len(df) * 100 / _prev_len:.2f}% of the rows!")
    y = df.sort_values(by=["time_started", "time_started"]).objective_to_minimize.values
    
    plt.clf()
    plt.plot(np.minimum.accumulate(y), label="HPO Trace", color="red")
    plt.hlines(
        np.percentile(y, reference),
        1,
        len(df),
        label="Reference score",
        color="black",
        linestyles="-"
    )
    print(f"Reference Score: {np.percentile(y, reference):.4f}; {100 * (1 - np.percentile(y, reference)):.4f}%")
    if dataset is not None:
        plt.title(f"{dataset}")
    plt.yscale("log")
    if ylims is not None:
        plt.ylim(*ylims)
    plt.legend(loc="lower left")
    plt.tight_layout()

    filename = "HPO.png" if filename is None else filename
    _plot_path = Path(summary_path).absolute() / filename
    plt.savefig(_plot_path, dpi=100)
    print(f"Saved at {_plot_path}")


def get_test_scores(
    data_path: Path,
    data_name: str,
    output_path: Path
) -> float:
    if not (output_path / "test_preds.npy").exists:
        return None

    test = pd.read_csv(data_path / data_name / "test.csv")
    labels = test["label"]

    with open(output_path / "test_preds.npy", "rb") as f:
        test_preds = np.load(f)

    return accuracy_score(labels, test_preds) * 100


def get_test_score_distribution(
    data_path: Path,
    data_name: str,
    root_directory: Path
):
    if (root_directory / "configs").exists():
        root_directory = root_directory / "configs"
    assert (root_directory /  "config_1_0").exists(), f"`root_directort` level is not correct!"

    scores = []
    for path in root_directory.iterdir():
        if not path.is_dir():
            continue
        try:
            with open(path / "score.yaml", "r") as f:
                _score = yaml.safe_load(f)["test_err"]
            scores.append((1 - _score) * 100)
        except Exception as e:
            pass
    
    print(min(scores))
    for p in [10, 25, 50, 75, 90]:
        print(np.percentile(scores, p))
    print(max(scores))
    print()
    print(f"Total entries: {len(scores)}")


if __name__ == "__main__":
    main()
# end of file