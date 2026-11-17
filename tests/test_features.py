"""The leakage rule: a horizon-h model may not see demand newer than h hours old."""

import re

import pytest

from features import HORIZONS, baseline_columns, feature_columns


def test_horizon_one_matches_the_original_feature_set():
    assert feature_columns(1, "recent") == [
        "pickup_lag_1h", "pickup_lag_2h", "pickup_lag_3h",
        "rolling_avg_3h_h1", "rolling_avg_24h_h1",
        "day_of_week", "is_weekend", "hour_of_day", "is_holiday",
    ]


@pytest.mark.parametrize("horizon", HORIZONS)
@pytest.mark.parametrize("feature_set", ["recent", "recent_plus_seasonal"])
def test_no_feature_is_newer_than_the_horizon(horizon, feature_set):
    for col in feature_columns(horizon, feature_set):
        lag = re.fullmatch(r"pickup_lag_(\d+)h", col)
        rolling = re.fullmatch(r"rolling_avg_\d+h_h(\d+)", col)
        if lag:
            assert int(lag.group(1)) >= horizon, f"{col} leaks into a {horizon}h forecast"
        if rolling:
            assert int(rolling.group(1)) >= horizon, f"{col} leaks into a {horizon}h forecast"


@pytest.mark.parametrize("horizon", HORIZONS)
def test_persistence_baseline_uses_the_latest_visible_hour(horizon):
    assert baseline_columns(horizon)["persistence"] == f"pickup_lag_{horizon}h"


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        feature_columns(4)
    with pytest.raises(ValueError):
        feature_columns(1, "everything")
