# 🧙‍♂️ DSLR – Logistic Regression from Scratch

## 🌟 Highlights

- **Machine Learning from Scratch** – No Scikit-Learn, only raw Python, NumPy, and Pandas.
- **One-vs-All Logistic Regression** – Classifies Hogwarts students into houses based on academic attributes.
- **Multiple Training Algorithms** – Supports **Gradient Descent, Stochastic Gradient Descent, and Mini-Batch Gradient Descent**.
- **Custom Feature Engineering & Standardization** – No built-in functions for mean, standard deviation, or statistics.
- **Data Analysis & Visualization** – Histograms, scatter plots, and pair plots for feature selection.
- **Command-Line Interface** – Train and predict using simple CLI commands.

---

## ℹ️ Overview

**DSLR (Data Science x Logistic Regression)** is a **machine learning project** that implements **logistic regression from scratch** to classify Hogwarts students into their respective houses. Unlike traditional ML projects, this one **avoids high-level ML libraries** and focuses on **raw algorithm implementation, feature selection, and optimization techniques**.

**Key Features:**\
✅ Supports multiple gradient descent techniques: **Batch, Stochastic, and Mini-Batch Gradient Descent**.\
✅ Implements a **one-vs-all classifier** for multi-class classification.\
✅ Uses **handwritten functions for mean, standard deviation, and normalization** (no built-in NumPy/Pandas shortcuts).\
✅ Provides **data visualization tools** to assist in feature selection and model interpretation.\
✅ Offers a **command-line interface** for training and prediction.

---

## 🚀 Installation & Setup

### **1. Clone the Repository**

```bash
git clone https://github.com/LuckyIntegral/dslr.git
cd dslr
```

### **2. Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## 📈 Training the Model

Run the training script, specifying the dataset and the gradient descent algorithm to use:

```bash
python src/logreg/logreg_train.py -d data/dataset_train.csv -a stochastic_gradient_descent
```

Supported algorithms:

- `gradient_descent`
- `stochastic_gradient_descent` (default)
- `mini_batch_gradient_descent`

This will generate a **weights file (**\`\`**)** that stores the trained model.

---

## 🔮 Making Predictions

Use the trained model to predict Hogwarts house classifications for a test dataset:

```bash
python src/logreg/logreg_predict.py -d data/dataset_test.csv -m weights.pkl
```

This will output a `houses.csv` file containing the predictions.

---

## 📊 Data Visualization

Before training, analyze the dataset using built-in visualization tools:

### **Generate a Histogram**

```bash
python src/plotting/histogram.py
```

### **Generate a Scatter Plot**

```bash
python src/plotting/scatter_plot.py
```

### **Generate a Pair Plot**

```bash
python src/plotting/pair_plot.py
```

---

## 🧪 Project Structure

```
.
├── data
│   ├── dataset_test.csv
│   └── dataset_train.csv
├── images (Generated plots)
├── requirements.txt
└── src
    ├── logreg
    │   ├── logistic_regression.py
    │   ├── logreg_train.py
    │   └── logreg_predict.py
    ├── plotting
    │   ├── histogram.py
    │   ├── pair_plot.py
    │   └── scatter_plot.py
    └── utils
        ├── describe.py (Basic dataset analysis)
```

---

## ✨ Future Improvements

- Implement **Regularization (L1/L2) to prevent overfitting**.
- Add **Cross-Validation & Learning Rate Scheduling**.
- Support **Real-Time Hyperparameter Tuning**.
- Implement a **Neural Network Version** for comparison.

---

## 🎓 Author

**Vitalii Frants**\
📍 42 Vienna – AI & Algorithms\
👉 [GitHub](https://github.com/LuckyIntegral)

---

### **🧙‍♂️ Ready to Sort Hogwarts Students? Try it now!**

