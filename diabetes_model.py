
"""Diabetes classification pipeline.

Educational project only; not intended for clinical diagnosis.
"""

from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler


MEASUREMENT_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]




def replace_measurement_zeros(X):
    """Treat zero measurements as missing for selected columns."""
    X = X.copy()

    for column in MEASUREMENT_COLUMNS:
        X[column] = X[column].astype("float64")
        X[column] = X[column].mask(X[column] == 0)

    return X

RANDOM_STATE = 42
DATA_PATH = Path(__file__).parent / "diabetes.csv"


def load_data():
    """Load the dataset and separate features from the target."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. "
            "Place diabetes.csv beside this script."
        )

    df = pd.read_csv(DATA_PATH)

    if "Outcome" not in df.columns:
        raise ValueError("Dataset must contain an 'Outcome' target column.")

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    return X, y



def build_pipeline():
    """Impute missing measurements, scale features, and fit logistic regression."""
    return Pipeline(
        steps=[
            (
                "zero_to_missing",
                FunctionTransformer(
                    replace_measurement_zeros,
                    validate=False,
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=RANDOM_STATE,
                    max_iter=2000,
                ),
            ),
        ]
    )


def main():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    pipeline = build_pipeline()

    param_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("Best parameters:", search.best_params_)
    print(f"Best training CV ROC-AUC: {search.best_score_:.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(f"Test ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()
