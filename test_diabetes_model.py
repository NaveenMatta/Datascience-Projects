
import numpy as np
import pandas as pd

from diabetes_model import (
    MEASUREMENT_COLUMNS,
    build_pipeline,
    load_data,
    replace_measurement_zeros,
)


def test_zero_handling_preserves_valid_zero_pregnancies():
    data = pd.DataFrame(
        {
            "Pregnancies": [0, 2],
            "Glucose": [0, 120],
            "BloodPressure": [70, 0],
            "SkinThickness": [20, 0],
            "Insulin": [0, 80],
            "BMI": [0.0, 25.0],
            "DiabetesPedigreeFunction": [0.2, 0.4],
            "Age": [25, 35],
        }
    )

    result = replace_measurement_zeros(data)

    assert result.loc[0, "Pregnancies"] == 0
    assert pd.isna(result.loc[0, "Glucose"])
    assert pd.isna(result.loc[1, "BloodPressure"])
    assert pd.isna(result.loc[0, "Insulin"])
    assert pd.isna(result.loc[0, "BMI"])
    assert result.loc[0, "DiabetesPedigreeFunction"] == 0.2


def test_pipeline_can_fit_and_predict():
    X, y = load_data()
    model = build_pipeline()
    model.fit(X.iloc[:100], y.iloc[:100])

    predictions = model.predict(X.iloc[100:110])

    assert len(predictions) == 10
    assert set(np.unique(predictions)).issubset({0, 1})


def test_measurement_columns_are_defined():
    assert "Glucose" in MEASUREMENT_COLUMNS
    assert "Insulin" in MEASUREMENT_COLUMNS
