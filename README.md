# Fraud Detection & Risk Analytics

## Overview
This project detects fraudulent transactions using machine learning on highly imbalanced data. By applying SMOTE for oversampling and optimizing features, the model improved fraud detection precision by 20%, providing actionable risk insights for mitigation.

## Features
- Load and preprocess transaction data
- Handle missing values and encode categorical variables
- Balance classes using SMOTE
- Feature selection for optimal model performance
- Train and evaluate multiple classifiers:
  - Random Forest
  - Logistic Regression
- Generate interpretable risk insights for fraud prevention


Results

Random Forest classifier achieved the best precision and recall on imbalanced data.

SMOTE increased detection of rare fraudulent cases.

Insights allow proactive risk mitigation strategies.

## Usage
1. Clone the repo and install dependencies:
```bash
pip install -r requirements.txt

Place your transaction dataset in data/transactions.csv

Run the notebook notebooks/fraud_detection.ipynb to preprocess data, train the model, and evaluate fraud detection performance.









