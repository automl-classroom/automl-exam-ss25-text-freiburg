# Text AutoML Template

This is a template for automatic text classification that supports multiple datasets and approaches.

## Project Structure

```
your_project/
├── src/
│   ├── __init__.py           # Package initialization
│   ├── automl.py             # TextAutoML class and models
│   └── datasets.py           # Dataset classes
├── data/                     # Data directory (created automatically)
│   ├── ag_news/
│   ├── imdb/
│   ├── amazon/
│   └── exam_text_dataset/
├── run_text.py              # Main execution script
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and prepare datasets:
```bash
# Quick setup (recommended)
python setup_data.py

# Or manually with Hugging Face datasets
python download_with_huggingface.py

# Or manually with direct downloads
python download_datasets.py
```

3. Verify data setup:
```bash
python verify_data.py
```

For detailed setup instructions, see [DATA_SETUP_GUIDE.md](DATA_SETUP_GUIDE.md).

## Usage

The system supports three datasets:
- `ag_news`: AG News (4-class news categorization)
- `imdb`: IMDB movie reviews (2-class sentiment)
- `amazon`: Amazon product reviews (5-class categorization)

### Basic Usage

```bash
# Run AG News classification
python run_text.py --dataset ag_news

# Run IMDB sentiment analysis
python run_text.py --dataset imdb

# Run Amazon reviews classification
python run_text.py --dataset amazon

# Specify output path and seed
python run_text.py --dataset ag_news --output-path my_predictions.npy --seed 123

# Run in quiet mode (less logging)
python run_text.py --dataset imdb --quiet
```

## Data Format

The system expects CSV files with the following format:
- `text`: The input text to classify
- `label`: The target class (integer from 0 to num_classes-1)

Example:
```csv
text,label
"This is a sample news article about world events",0
"Sports match between team A and team B",1
"Business earnings report from company X",2
"New technology breakthrough in AI",3
```

## Automatic Approach Selection

The system automatically selects the best approach based on dataset size:
- **Small datasets (< 500 samples)**: Bag-of-Words + Neural Network
- **Medium-small datasets (500-2000 samples)**: TF-IDF + Neural Network  
- **Medium datasets (2000-10000 samples)**: LSTM with embeddings
- **Medium-large datasets (10000-20000 samples)**: Fine-tuned pretrained models (DistilBERT)
- **Large datasets (> 20000 samples)**: Custom transformer architecture

See [APPROACHES_GUIDE.md](APPROACHES_GUIDE.md) for detailed explanations of each approach.

## Features

- **Multiple text processing approaches**: Bag-of-Words, TF-IDF, LSTM
- **Automatic preprocessing**: Lowercasing, punctuation removal, whitespace normalization
- **Dummy data generation**: Works even without real data files
- **GPU support**: Automatically uses CUDA if available
- **Reproducible results**: Seed-based random state control
- **Flexible architecture**: Easy to extend with new datasets and models

## Extending the System

### Adding a New Dataset

1. Create a new class in `src/datasets.py`:
```python
class MyDataset(BaseTextDataset):
    def get_num_classes(self) -> int:
        return 3  # Number of classes
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Load your data here
        pass
```

2. Add it to the choices in `run_text.py`

### Adding a New Model

1. Create a new model class in `src/automl.py`:
```python
class MyTextModel(nn.Module):
    def __init__(self, ...):
        # Initialize your model
        pass
    
    def forward(self, x):
        # Forward pass
        pass
```

2. Integrate it into the `TextAutoML` class

## Output

The system generates:
- **Predictions file**: numpy array with predicted class labels
- **Accuracy metrics**: If test labels are available
- **Classification report**: Detailed per-class metrics
- **Training logs**: Loss progression and model selection info

For the exam dataset, predictions are also saved to `data/exam_text_dataset/predictions.npy` for automatic grading.