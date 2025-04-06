# 🧙‍♂️ DSLR – The Hogwarts Sorting Hat Algorithm

## 🌟 Project Overview

This project recreates the Hogwarts Sorting Hat using machine learning! We've implemented a **logistic regression classifier from scratch** that can accurately assign students to their appropriate Hogwarts houses based on their academic performance.

### Key Features

- ✨ **Pure Magic (No High-Level Libraries)** - Everything is implemented from scratch with only basic Python, NumPy, and Pandas
- 🎯 **98%+ Accuracy** - Our model achieves McGonagall's required accuracy threshold
- 🔄 **Multiple Learning Algorithms** - Choose from three different gradient descent approaches
- 📊 **Data Visualization Suite** - Tools to understand Hogwarts student data patterns
- 🧰 **Custom Statistical Functions** - Hand-coded functions for all statistical operations

## 📊 Project Demonstrations

### 1. Data Analysis Tool

Our `describe.py` script provides statistical insights about the dataset, similar to Pandas' describe() function but built entirely from scratch:

```bash
python src/data/describe.py -d data/raw/dataset_train.csv
```

Add the `-b` flag for bonus statistics:

```bash
python src/data/describe.py -d data/raw/dataset_train.csv -b
```

### 2. Data Visualization

Explore the student data with various plots:

**Histogram**: Find courses with homogeneous distributions across houses
```bash
python src/visualization/histogram.py -c "Astronomy"
```

**Scatter Plot**: Discover correlated features
```bash
python src/visualization/scatter_plot.py -c "Astronomy" "Defense Against the Dark Arts"
```

**Pair Plot**: Comprehensive view of feature relationships
```bash
python src/visualization/pair_plot.py
```

### 3. The Sorting Hat Algorithm (Logistic Regression)

Train the sorting algorithm with different optimization methods:

```bash
# Default: Stochastic Gradient Descent (fastest)
python src/models/train.py -d data/processed/dataset_train.csv

# Batch Gradient Descent (most stable)
python src/models/train.py -d data/processed/dataset_train.csv -a gradient_descent

# Mini-Batch Gradient Descent (balanced approach)
python src/models/train.py -d data/processed/dataset_train.csv -a mini_batch_gradient_descent
```

Sort new students with the trained algorithm:

```bash
python src/models/predict.py -d data/processed/dataset_test.csv -m weights.pkl
```

## 🧠 The Magic Behind the Algorithm

This project implements a **one-vs-all logistic regression classifier** with three gradient descent optimization techniques:

1. **Batch Gradient Descent**: Updates weights using the entire dataset for each iteration
2. **Stochastic Gradient Descent**: Updates weights using one random example at a time
3. **Mini-Batch Gradient Descent**: Updates weights using small random batches of examples

Each approach offers different trade-offs between training speed and convergence stability.

## 🛠️ Setup Instructions

### Requirements

- Python 3.8+

### Installation

```bash
# Clone the repository
git clone https://github.com/LuckyIntegral/dslr.git
cd dslr

# Install dependencies
pip install -r requirements.txt

# Prepare the dataset (removes unnecessary features and handles missing values)
python src/data/prepare_dataset.py -i data/raw/dataset_train.csv -o data/processed/dataset_train.csv
python src/data/prepare_dataset.py -i data/raw/dataset_test.csv -o data/processed/dataset_test.csv
```

## 📚 Project Structure

```
dslr/
├── data/                      # Data directory
│   ├── raw/                   # Raw datasets
│   └── processed/             # Cleaned datasets
├── images/                    # Generated visualizations
├── notebooks/                 # Jupyter notebooks for demonstrations
├── src/                       # Source code
│   ├── data/                  # Data processing modules
│   │   ├── describe.py        # Statistical analysis tool
│   │   └── prepare_dataset.py # Data cleaning & preparation
│   ├── models/                # Machine learning models
│   │   ├── train.py           # Training algorithms
│   │   └── predict.py         # Prediction functions
│   └── visualization/         # Visualization tools
│       ├── histogram.py       # Histogram generator
│       ├── scatter_plot.py    # Scatter plot generator
│       └── pair_plot.py       # Pair plot generator
└── requirements.txt           # Project dependencies
```

## 🔮 Future Enhancements

- Add regularization to prevent overfitting
- Implement cross-validation for better model evaluation
- Create an interactive web application for real-time sorting
- Extend the algorithm to use neural networks for comparison

## 🧙‍♂️ Author

**Vitalii Frants**
📍 42 Vienna
👉 [GitHub](https://github.com/LuckyIntegral)
