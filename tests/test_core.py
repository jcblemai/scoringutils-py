import pytest
import pandas as pd
import numpy as np
from scoringutils_py.core import ForecastQuantile, ForecastPoint

@pytest.fixture
def sample_quantile_data():
    """Provides a sample DataFrame for testing quantile forecasts."""
    return pd.DataFrame({
        "observed": [10] * 5,
        "predicted": [8, 9, 10, 11, 12],
        "quantile_level": [0.1, 0.25, 0.5, 0.75, 0.9],
        "location": ["A"] * 5,
    })

@pytest.fixture
def forecast_unit():
    """Provides the forecast unit for testing."""
    return ["location"]

def test_forecast_quantile_creation(sample_quantile_data, forecast_unit):
    """Test that a ForecastQuantile object can be created successfully."""
    try:
        ForecastQuantile(sample_quantile_data, forecast_unit)
    except Exception as e:
        pytest.fail(f"ForecastQuantile creation failed with valid data: {e}")

def test_missing_columns_raises_error(sample_quantile_data, forecast_unit):
    """Test that missing required columns raise a ValueError."""
    for col in ["observed", "predicted", "quantile_level"]:
        with pytest.raises(ValueError, match=f"Required column '{col}' not found"):
            data = sample_quantile_data.drop(columns=col)
            ForecastQuantile(data, forecast_unit)

def test_missing_median_raises_error(sample_quantile_data, forecast_unit):
    """Test that data without a 0.5 quantile raises a ValueError."""
    data = sample_quantile_data[sample_quantile_data["quantile_level"] != 0.5]
    with pytest.raises(ValueError, match="Median forecast .* is required"):
        ForecastQuantile(data, forecast_unit)

def test_asymmetric_quantiles_raises_error(sample_quantile_data, forecast_unit):
    """Test that asymmetric quantiles raise a ValueError."""
    data = sample_quantile_data.copy()
    # Make quantiles asymmetric
    data.loc[data["quantile_level"] == 0.9, "quantile_level"] = 0.85
    with pytest.raises(ValueError, match="Asymmetric quantiles found"):
        fc = ForecastQuantile(data, forecast_unit)
        fc.score()

def test_score_method_returns_dataframe(sample_quantile_data, forecast_unit):
    """Test that the score method returns a pandas DataFrame with correct columns."""
    fc = ForecastQuantile(sample_quantile_data, forecast_unit)
    scores = fc.score()
    assert isinstance(scores, pd.DataFrame)
    assert "location" in scores.columns
    assert "wis" in scores.columns

def test_score_method_calculates_wis(sample_quantile_data, forecast_unit):
    """Test that the score method calculates a WIS score."""
    fc = ForecastQuantile(sample_quantile_data, forecast_unit)
    scores = fc.score()
    assert scores.shape[0] == 1 # One row for the single forecast unit
    wis_score = scores["wis"].iloc[0]
    assert isinstance(wis_score, (float, np.floating))
    assert wis_score >= 0

def test_median_only_forecast(forecast_unit):
    """Test that a forecast with only a median value can be scored."""
    data = pd.DataFrame({
        "observed": [10],
        "predicted": [12],
        "quantile_level": [0.5],
        "location": ["B"],
    })
    fc = ForecastQuantile(data, forecast_unit)
    scores = fc.score()
    assert "wis" in scores.columns
    # For median-only, WIS is just the absolute error
    assert np.isclose(scores["wis"].iloc[0], 2.0)


# ##################################
# # Tests for ForecastPoint
# ##################################

@pytest.fixture
def sample_point_data():
    """Provides a sample DataFrame for testing point forecasts."""
    return pd.DataFrame({
        "observed": [10, 20],
        "predicted": [12, 18],
        "location": ["A", "B"],
    })

def test_forecast_point_creation(sample_point_data, forecast_unit):
    """Test that a ForecastPoint object can be created successfully."""
    try:
        ForecastPoint(sample_point_data, forecast_unit)
    except Exception as e:
        pytest.fail(f"ForecastPoint creation failed with valid data: {e}")

def test_point_missing_columns_raises_error(sample_point_data, forecast_unit):
    """Test that missing required columns for point forecasts raise a ValueError."""
    for col in ["observed", "predicted"]:
        with pytest.raises(ValueError, match=f"Required column '{col}' not found"):
            data = sample_point_data.drop(columns=col)
            ForecastPoint(data, forecast_unit)

def test_point_score_method_returns_dataframe(sample_point_data, forecast_unit):
    """Test that the score method for point forecasts returns a DataFrame."""
    fc = ForecastPoint(sample_point_data, forecast_unit)
    scores = fc.score()
    assert isinstance(scores, pd.DataFrame)
    assert "location" in scores.columns
    assert "mae" in scores.columns
    assert len(scores) == 2

def test_point_score_method_calculates_mae(sample_point_data, forecast_unit):
    """Test that the score method for point forecasts calculates MAE correctly."""
    fc = ForecastPoint(sample_point_data, forecast_unit)
    scores = fc.score()

    # Check MAE for location A
    score_A = scores[scores["location"] == "A"]["mae"].iloc[0]
    assert np.isclose(score_A, 2.0) # |10 - 12| = 2

    # Check MAE for location B
    score_B = scores[scores["location"] == "B"]["mae"].iloc[0]
    assert np.isclose(score_B, 2.0) # |20 - 18| = 2
