# Credit Card Fraud Detection under Severe Class Imbalance

## Project Overview

This repository contains an exploratory data analysis (EDA) and feature selection pipeline for detecting fraudulent credit card transactions. Credit card fraud detection is a classic anomaly detection problem, often characterized by extreme class imbalance. The objective of this project is to analyze the data, understand the underlying distributions, and prepare the dataset for building a robust machine learning classification model that can accurately identify fraudulent transactions despite their rarity.

## 🔒 Important Note: PCA Encrypted Dataset

Due to strict privacy and confidentiality concerns surrounding financial data, this dataset is provided as a **PCA (Principal Component Analysis) encrypted dataset**. 

*   **Anonymized Features:** The original background information and feature names cannot be provided. The features `V1`, `V2`, ..., `V28` are the principal components obtained through a PCA transformation.
*   **Original Features:** The only features which have not been transformed with PCA are:
    *   **`Time`**: The seconds elapsed between each transaction and the first transaction in the dataset.
    *   **`Amount`**: The transaction amount.
*   **Target Variable:** The `Class` feature is the response variable, taking the value `1` in case of fraud and `0` for a legitimate transaction.

## 📊 Dataset Statistics

The dataset contains transactions made by European cardholders in September 2013 over two days. 

*   **Total Transactions:** 284,807
*   **Total Features:** 31 (including Time, Amount, and Class)
*   **Class Imbalance:** The dataset is highly unbalanced. The positive class (frauds) accounts for only **0.172%** of all transactions.

## 📂 Repository Contents

The project consists of the following Jupyter Notebooks:

1.  **`dataset_statistics.ipynb`**
    *   Comprehensive Exploratory Data Analysis (EDA).
    *   Statistical summaries of PCA components, `Time`, and `Amount`.
    *   Visualizations of the severe class imbalance.
    *   Analysis of transaction amount distributions (linear and logarithmic scales).
    *   Temporal analysis of transactions to find patterns between legitimate and fraudulent activities.

2.  **`feature_selection.ipynb`**
    *   Preliminary steps for data preprocessing and feature selection.
    *   Checking for null values and missing data.
    *   Re-verifying class distribution math and providing visual proof of the imbalance.
    *   Setting up the foundation for predictive modeling (e.g., using `RandomForestClassifier` from `sklearn`).

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `jupyter`

You can install the dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebooks

1.  Clone this repository to your local machine.
2.  Ensure that the `creditcard.csv` dataset is placed in the root directory of the project.
3.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
4.  Open `dataset_statistics.ipynb` or `feature_selection.ipynb` and run the cells sequentially.

## 🛠️ Technologies Used

*   **Python** - Core programming language
*   **Pandas & NumPy** - Data manipulation and numerical operations
*   **Matplotlib & Seaborn** - Data visualization
*   **Scikit-Learn** - Machine learning utilities

## 📈 Future Work

*   Implement techniques to handle class imbalance (e.g., SMOTE, ADASYN, or undersampling).
*   Train baseline anomaly detection models (Isolation Forest, Local Outlier Factor).
*   Train robust classifiers (Random Forest, XGBoost) and evaluate them using appropriate metrics like Precision-Recall AUC (PR-AUC) instead of standard Accuracy.