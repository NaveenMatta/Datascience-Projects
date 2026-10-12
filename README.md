# Diabetes Prediction Using Machine Learning


[![Python Tests](https://github.com/NaveenMatta/Datascience-Projects/actions/workflows/tests.yml/badge.svg)](https://github.com/NaveenMatta/Datascience-Projects/actions/workflows/tests.yml)


## Overview

An end-to-end machine learning project for diabetes classification using Python and Scikit-learn. The project demonstrates data preprocessing, missing-value handling, leakage-aware model training, hyperparameter tuning, cross-validation, and held-out test evaluation. Automated tests run through GitHub Actions on every push and pull request to the `main` branch.  

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

Both experiments use the same held-out test split. The alternative preprocessing increases positive-class recall from 0.500 to 0.704, identifying more positive cases, but also increases false positives from 15 to 25. It misses 16 positive cases, compared with 27 for the baseline. ROC-AUC decreases slightly from 0.820 to 0.813, so the trade-off depends on the evaluation objective.

These are results from a single test split, not evidence of clinical effectiveness. The model is for educational purposes only and must not be used for medical decisions.  

## Demo Output

The model selects hyperparameters using five-fold stratified cross-validation on the training set, then evaluates the selected pipeline on a held-out test set.

```text
Best parameters: {'model__C': 1.0, 'model__class_weight': 'balanced'}
Best training CV ROC-AUC: 0.845
Test accuracy: 0.734
Test ROC-AUC: 0.813

Confusion matrix:
[[75 25]
 [16 38]]
```

**Interpretation:** On this test split, the model identifies 38 of 54 positive cases and misses 16. It also flags 25 negative cases as positive. These are educational results, not evidence of clinical usefulness.

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
