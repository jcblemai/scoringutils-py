import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import scoringrules as sr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from functools import partial


# ##################################
# # Metric Implementation Helpers
# ##################################

def _interval_coverage(observed: float, lower: float, upper: float) -> int:
    """Check if observed value is within the interval."""
    return 1 if lower <= observed <= upper else 0

def _quantile_bias(observed: np.ndarray, predicted: np.ndarray, quantile_level: np.ndarray) -> float:
    """Calculate bias for a set of quantiles."""
    # I(y > q_alpha) - alpha
    errors = (observed > predicted) - quantile_level
    return np.mean(errors)


# ##################################
# # Get Default Metrics
# ##################################

def get_metrics_quantile() -> Dict[str, Any]:
    """
    Returns a dictionary of default scoring metrics for quantile forecasts.
    """
    # For interval coverage, the function is the same, the logic in score()
    # determines the interval range from the metric name.
    return {
        'wis': sr.weighted_interval_score,
        'bias': _quantile_bias,
        'interval_coverage_50': _interval_coverage,
        'interval_coverage_90': _interval_coverage,
    }


def get_metrics_point() -> Dict[str, Any]:
    """
    Returns a dictionary of default scoring metrics for point forecasts.
    """
    return {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
    }


# ##################################
# # Summarise Scores
# ##################################

def summarise_scores(scores: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    """
    Summarise scores by grouping and averaging.

    :param scores: A DataFrame of scores, typically the output of a `score()` method.
    :param by: A list of column names to group by.
    :return: A DataFrame with the summarised scores.
    """
    # Check if 'by' columns are in the scores DataFrame
    missing_cols = [col for col in by if col not in scores.columns]
    if missing_cols:
        raise ValueError(f"Grouping columns not found in scores DataFrame: {', '.join(missing_cols)}")

    # Identify metric columns (all numeric columns that are not in 'by')
    metric_cols = [
        col for col in scores.columns
        if pd.api.types.is_numeric_dtype(scores[col]) and col not in by
    ]

    if not metric_cols:
        # Return the grouping columns with no metrics if none are found
        return scores[by].drop_duplicates().reset_index(drop=True)

    summarised = scores.groupby(by)[metric_cols].mean().reset_index()
    return summarised


class Forecast:
    """Base class for forecast objects."""
    def __init__(self, data: pd.DataFrame, forecast_unit: List[str]):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input 'data' must be a pandas DataFrame.")
        self.data = data.copy()
        self.forecast_unit = forecast_unit
        self._validate_forecast_unit()

    def _validate_forecast_unit(self):
        """Check that the forecast unit columns are in the data."""
        missing_cols = [col for col in self.forecast_unit if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Columns specified in `forecast_unit` not found in data: {', '.join(missing_cols)}")

    def score(self, metrics: Optional[dict] = None) -> pd.DataFrame:
        """
        Score the forecasts. This method should be implemented by subclasses.
        """
        raise NotImplementedError("The 'score' method must be implemented by a subclass.")


class ForecastQuantile(Forecast):
    """Class for quantile-based forecasts."""
    def __init__(self, data: pd.DataFrame, forecast_unit: List[str]):
        super().__init__(data, forecast_unit)
        self._validate_quantile_data()

    def _validate_quantile_data(self):
        """Validate the specific requirements for quantile forecast data."""
        required_cols = ["observed", "predicted", "quantile_level"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in the data.")

        if not pd.api.types.is_numeric_dtype(self.data["quantile_level"]):
            raise TypeError("'quantile_level' column must be numeric.")
        if not ((self.data["quantile_level"] >= 0) & (self.data["quantile_level"] <= 1)).all():
            raise ValueError("'quantile_level' values must be between 0 and 1.")
        if 0.5 not in self.data["quantile_level"].unique():
            raise ValueError("Median forecast (quantile_level = 0.5) is required for WIS calculation.")

        if not pd.api.types.is_numeric_dtype(self.data["observed"]):
            raise TypeError("'observed' column must be numeric.")
        if not pd.api.types.is_numeric_dtype(self.data["predicted"]):
            raise TypeError("'predicted' column must be numeric.")

    def score(self, metrics: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculates scores for quantile forecasts.
        """
        if metrics is None:
            metrics = get_metrics_quantile()

        grouped = self.data.groupby(self.forecast_unit)
        results = []

        for name, group in grouped:
            result_row = dict(zip(self.forecast_unit, name if isinstance(name, tuple) else (name,)))
            observed = group["observed"].iloc[0]
            available_quantiles = group["quantile_level"].to_numpy()

            for metric_name, metric_func in metrics.items():
                # WIS calculation
                if metric_name == 'wis':
                    median_pred = group.loc[group["quantile_level"] == 0.5, "predicted"].iloc[0]
                    lower_quantiles = sorted([q for q in available_quantiles if q < 0.5])
                    if not lower_quantiles:
                        score_value = np.abs(observed - median_pred)
                    else:
                        upper_quantiles = sorted([q for q in available_quantiles if q > 0.5], reverse=True)
                        if len(lower_quantiles) != len(upper_quantiles) or not all(np.isclose(lq, 1 - uq) for lq, uq in zip(lower_quantiles, upper_quantiles)):
                             raise ValueError("WIS requires symmetric quantiles. Check your input data.")
                        lower_preds = group[group["quantile_level"].isin(lower_quantiles)].sort_values("quantile_level")["predicted"].to_numpy()
                        upper_preds = group[group["quantile_level"].isin(upper_quantiles)].sort_values("quantile_level", ascending=False)["predicted"].to_numpy()
                        alphas = 2 * np.array(lower_quantiles)
                        score_value = metric_func(observed, median_pred, lower_preds, upper_preds, alphas)
                    result_row[metric_name] = score_value

                # Bias calculation
                elif metric_name == 'bias':
                    score_value = metric_func(observed, group["predicted"].to_numpy(), available_quantiles)
                    result_row[metric_name] = score_value

                # Interval Coverage calculation
                elif 'interval_coverage' in metric_name:
                    try:
                        interval_range = int(metric_name.split('_')[-1])
                        alpha = (100 - interval_range) / 100
                        lower_q = alpha / 2
                        upper_q = 1 - alpha / 2

                        # Check if quantiles are available
                        if not (np.isclose(available_quantiles, lower_q).any() and np.isclose(available_quantiles, upper_q).any()):
                            score_value = np.nan # Not applicable
                        else:
                            lower_pred = group.loc[np.isclose(group["quantile_level"], lower_q), "predicted"].iloc[0]
                            upper_pred = group.loc[np.isclose(group["quantile_level"], upper_q), "predicted"].iloc[0]
                            score_value = metric_func(observed, lower_pred, upper_pred)
                        result_row[metric_name] = score_value
                    except (ValueError, IndexError):
                        # Handle cases where metric name is not in the expected format
                        result_row[metric_name] = np.nan

            results.append(result_row)

        return pd.DataFrame(results)


class ForecastPoint(Forecast):
    """Class for point forecasts."""
    def __init__(self, data: pd.DataFrame, forecast_unit: List[str]):
        super().__init__(data, forecast_unit)
        self._validate_point_data()

    def score(self, metrics: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Calculates scores for point forecasts.
        """
        if metrics is None:
            metrics = get_metrics_point()

        grouped = self.data.groupby(self.forecast_unit)
        results = []

        for name, group in grouped:
            result_row = dict(zip(self.forecast_unit, name if isinstance(name, tuple) else (name,)))
            observed = group["observed"].iloc[0]
            predicted = group["predicted"].iloc[0]

            for metric_name, metric_func in metrics.items():
                score_value = metric_func([observed], [predicted])
                result_row[metric_name] = score_value

            results.append(result_row)

        if not results:
            return pd.DataFrame()

        return pd.DataFrame(results)


    def _validate_point_data(self):
        """Validate the specific requirements for point forecast data."""
        required_cols = ["observed", "predicted"]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in the data.")

        if not pd.api.types.is_numeric_dtype(self.data["observed"]):
            raise TypeError("'observed' column must be numeric.")
        if not pd.api.types.is_numeric_dtype(self.data["predicted"]):
            raise TypeError("'predicted' column must be numeric.")
