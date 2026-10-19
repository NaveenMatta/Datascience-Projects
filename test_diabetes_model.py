
import numpy as np
import pandas as pd
import pytest

import diabetes_model as dm


def make_test_data(n=40):
    """Create deterministic data without depending on the CSV file."""
    return pd.DataFrame(
        {
            "Pregnancies": np.arange(n) % 6,
            "Glucose": 90 + np.arange(n) % 80,
            "BloodPressure": 60 + np.arange(n) % 30,
            "SkinThickness": 15 + np.arange(n) % 25,
            "Insulin": 50 + np.arange(n) % 100,
            "BMI": 20 + (np.arange(n) % 15),
            "DiabetesPedigreeFunction": 0.1 + (np.arange(n) % 10) / 10,
            "Age": 20 + np.arange(n) % 50,
        }
    )


def test_zero_handling_preserves_valid_zero_pregnancies():
    data = make_test_data(2)
    data.loc[0, "Pregnancies"] = 0
    data.loc[0, "Glucose"] = 0
    data.loc[1, "BloodPressure"] = 0
    data.loc[0, "SkinThickness"] = 0
    data.loc[0, "Insulin"] = 0
    data.loc[0, "BMI"] = 0

    result = dm.replace_measurement_zeros(data)

    assert result.loc[0, "Pregnancies"] == 0

    for column in dm.MEASUREMENT_COLUMNS:
        assert pd.isna(result.loc[0, column])

    assert pd.isna(result.loc[1, "BloodPressure"])


def test_zero_handling_does_not_modify_input():
    data = make_test_data(3)
    data.loc[0, "Glucose"] = 0
    original = data.copy(deep=True)

    dm.replace_measurement_zeros(data)

    pd.testing.assert_frame_equal(data, original)


def test_pipeline_can_fit_and_predict():
    X = make_test_data(40)
    y = pd.Series([0, 1] * 20)

    model = dm.build_pipeline()
    model.fit(X.iloc[:30], y.iloc[:30])

    predictions = model.predict(X.iloc[30:])

    assert len(predictions) == 10
    assert set(np.unique(predictions)).issubset({0, 1})


def test_measurement_columns_are_defined():
    expected = {
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    }

    assert set(dm.MEASUREMENT_COLUMNS) == expected


def test_load_data_raises_when_file_is_missing(
    monkeypatch, tmp_path
):
    missing_path = tmp_path / "missing.csv"
    monkeypatch.setattr(dm, "DATA_PATH", missing_path)

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        dm.load_data()


def test_load_data_raises_when_outcome_is_missing(
    monkeypatch, tmp_path
):
    csv_path = tmp_path / "invalid.csv"
    make_test_data(5).to_csv(csv_path, index=False)

    monkeypatch.setattr(dm, "DATA_PATH", csv_path)

    with pytest.raises(ValueError, match="Outcome"):
        dm.load_data()
