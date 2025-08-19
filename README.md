# scoringutils_py

A Python package for evaluating and scoring forecasts, inspired by the R package `scoringutils`.

This package provides tools to evaluate forecasts in a convenient framework based on pandas DataFrames.

## Current Status

This package is an initial conversion of the original R `scoringutils` package. It currently supports:
- **Quantile Forecasts**: Scored with Weighted Interval Score (WIS), Bias, and Interval Coverage.
- **Point Forecasts**: Scored with Mean Absolute Error (MAE) and Mean Squared Error (MSE).
- **Score Summarisation**: Scores can be summarised by grouping variables.

Many features from the R package (e.g., other forecast types, plotting) have not yet been implemented.

## Installation

To install the package, clone the repository and install it using pip:

```bash
git clone <repository_url>
cd <repository_name>
pip install .
```

## Quick Start

Here is a simple example of how to use `scoringutils_py` to score a quantile forecast.

### 1. Prepare your data

Your forecast data should be in a pandas DataFrame with the following columns:
- `observed`: The true observed value.
- `predicted`: The predicted value for a given quantile.
- `quantile_level`: The quantile level (between 0 and 1).
- Columns that uniquely identify a single forecast (the `forecast_unit`).

```python
import pandas as pd
from scoringutils_py.core import ForecastQuantile

# Create sample data
data = pd.DataFrame({
    "observed": [10] * 5,
    "predicted": [8, 9, 10, 11, 12],
    "quantile_level": [0.1, 0.25, 0.5, 0.75, 0.9],
    "location": ["A"] * 5,
})

# Define the unit of a single forecast
forecast_unit = ["location"]
```

### 2. Create a Forecast Object

Use the `ForecastQuantile` class to validate and represent your forecast data.

```python
fc = ForecastQuantile(data, forecast_unit)
```

### 3. Score the Forecasts

Use the `score` method to calculate scores. By default, it calculates a range of relevant metrics.

```python
scores = fc.score()

print(scores)
```

This will output a DataFrame with the calculated scores for each forecast unit.
```
  location       wis      bias  interval_coverage_50  interval_coverage_90
0        A  1.234567 -0.055555                     1                     1 # Example scores
```

### 4. Summarise Scores

You can summarise the raw scores by one or more grouping variables using the `summarise_scores` function.

```python
from scoringutils_py.core import summarise_scores

# Add a model column to the data for summarisation
scores['model'] = 'MyModel'

summary = summarise_scores(scores, by=['model'])

print(summary)
```

This will output the mean of each score, grouped by the `by` columns.
```
    model       wis      bias  interval_coverage_50  interval_coverage_90
0  MyModel  1.234567 -0.055555                     1                     1
```

---

### Scoring Point Forecasts

Here is an example of how to score point forecasts.

#### 1. Prepare your data

For point forecasts, your DataFrame needs `observed` and `predicted` columns, along with the `forecast_unit`.

```python
import pandas as pd
from scoringutils_py.core import ForecastPoint

# Create sample data
data = pd.DataFrame({
    "observed": [10, 20],
    "predicted": [12, 18],
    "location": ["A", "B"],
})

# Define the unit of a single forecast
forecast_unit = ["location"]
```

#### 2. Create a Forecast Object

Use the `ForecastPoint` class.

```python
fc = ForecastPoint(data, forecast_unit)
```

#### 3. Score the Forecasts

The `score` method for point forecasts calculates Mean Absolute Error (MAE) and Mean Squared Error (MSE) by default.

```python
scores = fc.score()

print(scores)
```

This will output:
```
  location  mae  mse
0        A  2.0  4.0
1        B  2.0  4.0
```
