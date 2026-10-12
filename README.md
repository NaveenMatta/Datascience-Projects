# Diabetes Prediction Using Machine Learning

## Overview
An educational binary classification project using the Pima Indians Diabetes dataset. The project compares a baseline Logistic Regression pipeline with an alternative preprocessing experiment.

**Disclaimer:** This project is for educational purposes only and is not intended for clinical diagnosis.

## Tech Stack
- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Data preprocessing and model evaluation

## Project Structure
- `diabetes.csv` — dataset
- `diabetes_model.py` — training and evaluation pipeline
- `requirements.txt` — Python dependencies
- `model_results.txt` — baseline results
- `improved_model_results.txt` — alternative preprocessing results
- `Diabetes prediction using ML.ipynb` — original notebook

## Methodology
- Stratified 80/20 train-test split
- Five-fold stratified cross-validation on the training set
- Median imputation and RobustScaler preprocessing
- Logistic Regression hyperparameter tuning using ROC-AUC
- Evaluation using accuracy, ROC-AUC, precision, recall, F1-score, and confusion matrix

The alternative experiment treats zero values as missing in Glucose, BloodPressure, SkinThickness, Insulin, and BMI. This is an assumption that should be evaluated, not treated as ground truth.

## Results

| Metric | Baseline | Alternative preprocessing |
|---|---:|---:|
| Test accuracy | 0.727 | 0.734 |
| Test ROC-AUC | 0.820 | 0.813 |
| Positive-class recall | 0.500 | 0.704 |
| Positive-class F1-score | 0.56 | 0.65 |

Both runs use the same held-out test split. The alternative experiment detects more positive cases but also produces more false positives. Its ROC-AUC is slightly lower, so it is not better on every metric.

These results come from one split and are not evidence of clinical usefulness.

## How to Run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the pipeline:

```bash
python diabetes_model.py
```

Ensure `diabetes.csv` is in the same directory as `diabetes_model.py`.

## Limitations and Future Work
- Investigate dataset documentation and zero-value assumptions.
- Evaluate variability using repeated or nested cross-validation.
- Examine threshold selection and precision-recall trade-offs.
- Add automated tests and a reproducible environment.
- Do not use this model for medical decisions.
