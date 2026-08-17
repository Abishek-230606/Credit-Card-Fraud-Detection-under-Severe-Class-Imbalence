# Credit Card Fraud Detection under Severe Class Imbalance

## Project Overview

This repository contains an exploratory data analysis (EDA), feature selection pipeline, and mathematical model implementations for detecting fraudulent credit card transactions. Credit card fraud detection is a classic anomaly detection problem, often characterized by extreme class imbalance. The objective of this project is to analyze the data, understand the underlying distributions, and build a robust machine learning classification model that can accurately identify fraudulent transactions despite their rarity.

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

The project has been refactored into distinct modules:

### 1. Dataset Statistics and Feature Selection (`/dataset_statistics_and_feature_selection`)
*   **`dataset_statistics.ipynb`**: Comprehensive Exploratory Data Analysis (EDA). Statistical summaries of PCA components, visualizations of the severe class imbalance, and temporal analysis of transactions.
*   **`feature_selection.ipynb`**: Preliminary steps for data preprocessing and feature selection. Provides correlation heatmaps and sets up the foundation for predictive modeling.
*   **`graphs_of_dataset_statistics/poster.ipynb`**: Generates key high-quality visualizations (Bar Charts, Histograms, Boxplots, Scatter Plots, and Heatmaps) used for presentation posters.

### 2. Mathematical Models (`/mathematical_model`)
*   **`mathematical_model.ipynb`**: Implementation of techniques to handle class imbalance directly from scratch. Includes a custom mathematical implementation of **SMOTE** (Synthetic Minority Over-sampling Technique) and **Cost-Sensitive Logistic Regression** utilizing weighted Log Loss to penalize missed frauds.
*   **`gpu_math_model.ipynb`**: A GPU-accelerated version of the mathematical model using **CuPy** to run Gradient Descent on Nvidia GPUs, falling back to NumPy (CPU) gracefully if CUDA libraries are unavailable. Features fully vectorized CPU SMOTE generation.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the following libraries:
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `seaborn`
*   `scikit-learn`
*   `jupyter`
*   `cupy` (optional, for GPU acceleration in the math model)

You can install the dependencies using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Running the Notebooks

1.  Clone this repository to your local machine.
2.  Ensure that the `creditcard.csv` dataset is placed inside the `dataset_statistics_and_feature_selection/` directory.
3.  Start the Jupyter Notebook server:
    ```bash
    jupyter notebook
    ```
4.  Navigate to the respective directories to open and run the notebooks sequentially.

## 🛠️ Technologies Used

*   **Python** - Core programming language
*   **Pandas & NumPy** - Data manipulation and numerical operations
*   **Matplotlib & Seaborn** - Data visualization
*   **Scikit-Learn** - Machine learning utilities
*   **CuPy** - GPU-accelerated computing

## 📈 Future Work

*   Train and compare other baseline anomaly detection models (Isolation Forest, Local Outlier Factor).
*   Train robust classifiers (Random Forest, XGBoost) and evaluate them using appropriate metrics like Precision-Recall AUC (PR-AUC) alongside our custom Cost-Sensitive Logistic Regression.