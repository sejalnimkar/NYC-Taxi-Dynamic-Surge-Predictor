"""
src/evaluate.py

Forecast error metrics and train/validation/holdout splitting.
Pure pandas/numpy so it can be unit-tested without a warehouse.
"""

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """
    RMSE  - penalises big misses; dominated by the busiest zones.
    MAE   - average miss in trips, easier to read.
    WAPE  - total absolute error / total demand. Scale-free, and unlike MAPE
            it does not blow up on the many zone-hours with zero trips.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("cannot score an empty set")
    err = y_true - y_pred
    total = np.abs(y_true).sum()
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "wape": float(np.abs(err).sum() / total) if total > 0 else float("nan"),
    }


def split_by_time(df: pd.DataFrame, holdout_start, val_fraction: float = 0.2, ts_col="pickup_hour_ts"):
    """
    Three time-ordered pieces, never shuffled:
      train   - earliest (1 - val_fraction) of the hours before holdout_start
      val     - the remaining hours before holdout_start (for tuning)
      holdout - everything from holdout_start on (touched once, for the final score)

    Splitting on distinct hours (not rows) guarantees no hour sits on both sides.
    """
    holdout_start = pd.Timestamp(holdout_start)
    before = df[df[ts_col] < holdout_start]
    holdout = df[df[ts_col] >= holdout_start]

    hours = np.sort(before[ts_col].unique())
    if len(hours) < 2:
        raise ValueError("need at least two distinct hours before holdout_start")
    cut = hours[int(len(hours) * (1 - val_fraction)) - 1]
    train = before[before[ts_col] <= cut]
    val = before[before[ts_col] > cut]
    return train, val, holdout


def metrics_by_group(df: pd.DataFrame, group_col: str, y_col: str, pred_col: str) -> pd.DataFrame:
    """Error metrics per group (e.g. per borough), with row counts and mean demand."""
    rows = []
    for key, part in df.groupby(group_col):
        rows.append(
            {
                group_col: key,
                "rows": len(part),
                "mean_actual": float(part[y_col].mean()),
                **regression_metrics(part[y_col], part[pred_col]),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)
