# Diabetes Prediction Using Machine Learning


[![Python Tests](https://github.com/NaveenMatta/Datascience-Projects/actions/workflows/tests.yml/badge.svg)](https://github.com/NaveenMatta/Datascience-Projects/actions/workflows/tests.yml)


## Overview

An educational binary classification project using the Pima Indians Diabetes dataset. It compares a baseline Logistic Regression pipeline with an alternative preprocessing experiment.

**Disclaimer:** This project is for educational purposes only and is not intended for clinical diagnosis.

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Pytest
- GitHub Actions

## Project Structure

- `data/diabetes.csv` — dataset
- `diabetes_model.py` — training and evaluation pipeline
- `test_diabetes_model.py` — automated tests
- `requirements.txt` — Python dependencies
- `Diabetes prediction using ML.ipynb` — original exploratory notebook
- `Ecommerce customer purchase intention mod...` — original exploratory notebook
- `ML Project using LSTM-checkpoint.ipynb` — original LSTM notebook
- `ML Project using ML-checkpoint.ipynb` — original ML notebook
- `.github/workflows/tests.yml` — GitHub Actions CI workflow

## Methodology

- Stratified 80/20 train-test split
- Five-fold stratified cross-validation on the training set
- Median imputation and RobustScaler preprocessing
- Logistic Regression hyperparameter tuning using ROC-AUC
- Evaluation using accuracy, ROC-AUC, precision, recall, F1-score, and confusion matrix

The alternative experiment treats zero values as missing in Glucose, BloodPressure, SkinThickness, Insulin, and BMI. This is an assumption that requires further validation.

## Results

| Metric | Baseline | Alternative preprocessing |
|---|---:|---:|
| Test accuracy | 0.727 | 0.734 |
| Test ROC-AUC | 0.820 | 0.813 |
| Positive-class recall | 0.500 | 0.704 |
| Positive-class F1-score | 0.56 | 0.65 |

Both experiments use the same held-out test split. The alternative detects more positive cases but also produces more false positives. Its ROC-AUC is slightly lower, so it is not better on every metric.

These results come from one split and do not establish clinical usefulness.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the model from the project root:

```bash
python diabetes_model.py
```

Run automated tests:

```bash
python -m pytest test_diabetes_model.py -v
```

The dataset is expected at `data/diabetes.csv` relative to the project root.

## Limitations and Future Work

- Validate the zero-value assumptions against dataset documentation.
- Evaluate variability using repeated or nested cross-validation.
- Investigate threshold selection and precision-recall trade-offs.
- Expand automated tests and improve environment reproducibility.
- Do not use this model for medical decisions.  
