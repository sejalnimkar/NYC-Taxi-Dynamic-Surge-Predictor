import numpy as np
import pandas as pd
import pytest

from evaluate import metrics_by_group, regression_metrics, split_by_time


def test_regression_metrics_known_values():
    m = regression_metrics([10, 0, 20], [8, 1, 24])
    assert m["mae"] == pytest.approx(7 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(21 / 3))
    assert m["wape"] == pytest.approx(7 / 30)


def test_wape_is_nan_not_an_error_when_demand_is_all_zero():
    assert np.isnan(regression_metrics([0, 0], [0, 1])["wape"])


def test_metrics_reject_bad_input():
    with pytest.raises(ValueError):
        regression_metrics([], [])
    with pytest.raises(ValueError):
        regression_metrics([1, 2], [1])


def _grid(days=10, zones=3):
    hours = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    return pd.DataFrame(
        [(z, h, 1) for z in range(zones) for h in hours], columns=["zone_id", "pickup_hour_ts", "pickup_count"]
    )


def test_split_by_time_is_ordered_and_does_not_overlap():
    df = _grid()
    train, val, holdout = split_by_time(df, "2026-01-09", val_fraction=0.25)
    assert train["pickup_hour_ts"].max() < val["pickup_hour_ts"].min()
    assert val["pickup_hour_ts"].max() < holdout["pickup_hour_ts"].min()
    assert holdout["pickup_hour_ts"].min() == pd.Timestamp("2026-01-09")
    assert len(train) + len(val) + len(holdout) == len(df)
    # 8 days before the holdout, 25% validation -> 6 days train, 2 days val, for every zone
    assert train["pickup_hour_ts"].nunique() == 6 * 24
    assert val["pickup_hour_ts"].nunique() == 2 * 24


def test_split_never_puts_one_hour_on_both_sides():
    train, val, _ = split_by_time(_grid(), "2026-01-09")
    assert set(train["pickup_hour_ts"]).isdisjoint(set(val["pickup_hour_ts"]))


def test_metrics_by_group():
    df = pd.DataFrame({"g": ["a", "a", "b"], "y": [10, 10, 4], "p": [8, 12, 4]})
    out = metrics_by_group(df, "g", "y", "p").set_index("g")
    assert out.loc["a", "rows"] == 2 and out.loc["a", "mae"] == pytest.approx(2.0)
    assert out.loc["b", "rmse"] == 0
