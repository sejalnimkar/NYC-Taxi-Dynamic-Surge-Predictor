"""
src/features.py

Which columns of fct_zone_features a model may use at each forecast horizon.

A forecast made h hours ahead can only use demand observed up to h hours
before the target hour. Keeping that rule in one place (rather than in a
hand-typed list per model) is what stops a 3-hour-ahead model from quietly
using the 1-hour lag.
"""

HORIZONS = (1, 2, 3)
TARGET = "pickup_count"

# Describe the target hour itself; all known in advance.
CALENDAR_FEATURES = ["day_of_week", "is_weekend", "hour_of_day", "is_holiday"]

# Same hour yesterday / last week. Older than any horizon here, so always usable.
SEASONAL_FEATURES = ["pickup_lag_24h", "pickup_lag_168h"]

FEATURE_SETS = ("recent", "recent_plus_seasonal")


def feature_columns(horizon: int, feature_set: str = "recent_plus_seasonal") -> list[str]:
    """Columns available when forecasting `horizon` hours ahead."""
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}, got {horizon}")
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set}")

    lags = [f"pickup_lag_{horizon + i}h" for i in range(3)]
    rolling = [f"rolling_avg_3h_h{horizon}", f"rolling_avg_24h_h{horizon}"]
    columns = lags + rolling + CALENDAR_FEATURES
    if feature_set == "recent_plus_seasonal":
        columns += SEASONAL_FEATURES
    return columns


def baseline_columns(horizon: int) -> dict[str, str]:
    """Model-free forecasts to beat, as {name: column holding the forecast}."""
    return {
        "persistence": f"pickup_lag_{horizon}h",  # "same as the latest hour we have seen"
        "same_hour_yesterday": "pickup_lag_24h",
        "same_hour_last_week": "pickup_lag_168h",
    }
