import pytest
import pandas as pd
import numpy as np
from scoringutils_py.core import (
    ForecastQuantile, ForecastPoint, summarise_scores,
    get_metrics_quantile, get_metrics_point
)

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
    """Test that asymmetric quantiles raise a ValueError for WIS calculation."""
    data = sample_quantile_data.copy()
    # Make quantiles asymmetric
    data.loc[data["quantile_level"] == 0.9, "quantile_level"] = 0.85
    with pytest.raises(ValueError, match="WIS requires symmetric quantiles"):
        fc = ForecastQuantile(data, forecast_unit)
        # We only test the wis metric, as others might not require symmetry
        fc.score(metrics={'wis': get_metrics_quantile()['wis']})

def test_score_method_returns_dataframe(sample_quantile_data, forecast_unit):
    """Test that the score method returns a pandas DataFrame with correct columns."""
    fc = ForecastQuantile(sample_quantile_data, forecast_unit)
    scores = fc.score()
    assert isinstance(scores, pd.DataFrame)
    assert "location" in scores.columns
    # Check for all default quantile metrics
    expected_metrics = ["wis", "bias", "interval_coverage_50", "interval_coverage_90"]
    for metric in expected_metrics:
        assert metric in scores.columns

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
    assert "bias" in scores.columns
    # For median-only, WIS is just the absolute error
    assert np.isclose(scores["wis"].iloc[0], 2.0)
    # Check that coverage is NaN as intervals are not present
    assert np.isnan(scores["interval_coverage_50"].iloc[0])


def test_quantile_bias_calculation(forecast_unit):
    """Test the quantile bias calculation."""
    data = pd.DataFrame({
        "observed": [10] * 3,
        "predicted": [8, 10, 12],
        "quantile_level": [0.25, 0.5, 0.75],
        "location": ["C"] * 3,
    })
    fc = ForecastQuantile(data, forecast_unit)
    scores = fc.score(metrics={'bias': get_metrics_quantile()['bias']})
    # I(10 > 8) - 0.25 = 1 - 0.25 = 0.75
    # I(10 > 10) - 0.5 = 0 - 0.5 = -0.5
    # I(10 > 12) - 0.75 = 0 - 0.75 = -0.75
    # mean = (0.75 - 0.5 - 0.75) / 3 = -0.5 / 3
    assert np.isclose(scores["bias"].iloc[0], -0.5 / 3)

def test_interval_coverage_calculation(forecast_unit):
    """Test the interval coverage calculation."""
    data = pd.DataFrame({
        "observed": [10] * 5,
        "predicted": [8, 9, 10, 11, 12],
        "quantile_level": [0.05, 0.25, 0.5, 0.75, 0.95],
        "location": ["D"] * 5,
    })
    fc = ForecastQuantile(data, forecast_unit)
    metrics_to_test = {
        'interval_coverage_50': get_metrics_quantile()['interval_coverage_50'],
        'interval_coverage_90': get_metrics_quantile()['interval_coverage_90'],
    }
    scores = fc.score(metrics=metrics_to_test)
    # 50% interval is [9, 11], 10 is inside -> coverage = 1
    assert scores["interval_coverage_50"].iloc[0] == 1
    # 90% interval is [8, 12], 10 is inside -> coverage = 1
    assert scores["interval_coverage_90"].iloc[0] == 1


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
    assert "mse" in scores.columns
    assert len(scores) == 2

def test_point_score_method_calculates_mae(sample_point_data, forecast_unit):
    """Test that the score method for point forecasts calculates MAE correctly."""
    fc = ForecastPoint(sample_point_data, forecast_unit)
    scores = fc.score(metrics={'mae': get_metrics_point()['mae']})

    # Check MAE for location A
    score_A = scores[scores["location"] == "A"]["mae"].iloc[0]
    assert np.isclose(score_A, 2.0) # |10 - 12| = 2

    # Check MAE for location B
    score_B = scores[scores["location"] == "B"]["mae"].iloc[0]
    assert np.isclose(score_B, 2.0) # |20 - 18| = 2

def test_point_score_method_calculates_mse(sample_point_data, forecast_unit):
    """Test that the score method for point forecasts calculates MSE correctly."""
    fc = ForecastPoint(sample_point_data, forecast_unit)
    scores = fc.score(metrics={'mse': get_metrics_point()['mse']})

    # Check MSE for location A
    score_A = scores[scores["location"] == "A"]["mse"].iloc[0]
    assert np.isclose(score_A, 4.0) # (10 - 12)^2 = 4

    # Check MSE for location B
    score_B = scores[scores["location"] == "B"]["mse"].iloc[0]
    assert np.isclose(score_B, 4.0) # (20 - 18)^2 = 4


# ##################################
# # Tests for summarise_scores
# ##################################

def test_summarise_scores():
    """Test the summarise_scores function."""
    scores = pd.DataFrame({
        "model": ["A", "A", "B", "B"],
        "location": ["L1", "L2", "L1", "L2"],
        "wis": [10, 20, 30, 40],
        "mae": [1, 2, 3, 4],
    })

    # Summarise by model
    summary = summarise_scores(scores, by=["model"])
    assert len(summary) == 2
    assert np.isclose(summary[summary["model"] == "A"]["wis"].iloc[0], 15) # mean(10, 20)
    assert np.isclose(summary[summary["model"] == "A"]["mae"].iloc[0], 1.5) # mean(1, 2)
    assert np.isclose(summary[summary["model"] == "B"]["wis"].iloc[0], 35) # mean(30, 40)
    assert np.isclose(summary[summary["model"] == "B"]["mae"].iloc[0], 3.5) # mean(3, 4)

    # Summarise by location
    summary_loc = summarise_scores(scores, by=["location"])
    assert len(summary_loc) == 2
    assert np.isclose(summary_loc[summary_loc["location"] == "L1"]["wis"].iloc[0], 20) # mean(10, 30)
