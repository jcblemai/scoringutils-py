import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
import scoringrules as sr


def get_metrics_quantile() -> Dict[str, Any]:
    """
    Returns a dictionary of default scoring metrics for quantile forecasts.
    """
    return {
        'wis': sr.weighted_interval_score,
    }


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

            # Prepare data for scoring functions
            median_pred = group.loc[group["quantile_level"] == 0.5, "predicted"].iloc[0]
            lower_quantiles = sorted([q for q in group["quantile_level"].unique() if q < 0.5])
            upper_quantiles = sorted([q for q in group["quantile_level"].unique() if q > 0.5], reverse=True)

            if len(lower_quantiles) != len(upper_quantiles):
                raise ValueError("Quantile forecasts must be symmetric around the median for WIS calculation.")
            # upper_quantiles are sorted descending, so we don't need to reverse
            for lq, uq in zip(lower_quantiles, upper_quantiles):
                if not np.isclose(lq, 1 - uq):
                    raise ValueError(f"Asymmetric quantiles found: {lq} and {uq}.")

            # Calculate all requested metrics
            for metric_name, metric_func in metrics.items():
                if metric_name == 'wis':
                    if not lower_quantiles:
                        # Median-only case, WIS is the absolute error
                        score_value = np.abs(observed - median_pred)
                    else:
                        lower_preds = group[group["quantile_level"].isin(lower_quantiles)].sort_values("quantile_level")["predicted"].to_numpy()
                        upper_preds = group[group["quantile_level"].isin(upper_quantiles)].sort_values("quantile_level", ascending=False)["predicted"].to_numpy()
                        alphas = 2 * np.array(lower_quantiles)
                        score_value = metric_func(
                            observed,
                            median_pred,
                            lower_preds,
                            upper_preds,
                            alphas
                        )
                    result_row[metric_name] = score_value
                # In the future, other metrics could be handled here
                # with different data preparation logic.

            results.append(result_row)

        return pd.DataFrame(results)
