# Diabetes Prediction Using Machine Learning

## Overview
An educational binary classification project that predicts the likelihood of diabetes using medical features from a tabular dataset.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Machine Learning
- Logistic Regression
- Data preprocessing and model evaluation

## Project Files
- `Diabetes prediction using ML.ipynb` — original exploratory notebook
- `diabetes_model.py` — reusable machine learning pipeline
- `diabetes.csv` — dataset used by the pipeline

## Key Implementation Details
- Stratified train-test split
- Missing-value imputation
- Robust feature scaling
- Logistic Regression with hyperparameter tuning
- Five-fold stratified cross-validation
- Evaluation using accuracy, ROC-AUC, classification report, and confusion matrix

## How to Run

Install the dependencies:

```bash
pip install pandas scikit-learn
```

Run the model:

```bash
python diabetes_model.py
```

Ensure `diabetes.csv` is in the same directory as `diabetes_model.py`.

## Limitations
This project is for educational purposes only and is not intended for clinical diagnosis. Model performance must be evaluated carefully before any real-world use.
