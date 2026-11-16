"""
src/train.py

Trains one LightGBM model per forecast horizon (1, 2 and 3 hours ahead) on
fct_zone_features, and scores each against model-free baselines on a
held-out period the models never saw.

For every horizon it reports, on the holdout:
  - three baselines (persistence, same hour yesterday, same hour last week)
  - LightGBM on the recent-demand features
  - LightGBM with the daily/weekly seasonal lags added
Results go to MLflow and to reports/metrics.json + reports/holdout_metrics.md.

Usage:
    python src/train.py                               # Snowflake, default params
    python src/train.py --n-trials 30                 # re-tune with Optuna first
    python src/train.py --source duckdb --duckdb-path nyc_taxi.duckdb \
        --holdout-start 2026-01-13 --no-mlflow        # local / CI run

Called by Airflow's `train_model` task via docker exec.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate import metrics_by_group, regression_metrics, split_by_time  # noqa: E402
from features import FEATURE_SETS, HORIZONS, TARGET, baseline_columns, feature_columns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ZONE_LOOKUP = REPO_ROOT / "nyc_surge_predictor" / "seeds" / "taxi_zone_lookup.csv"
REPORTS_DIR = REPO_ROOT / "reports"

EXPERIMENT_NAME = "nyc_surge_gbdt"
DEFAULT_HOLDOUT_START = "2026-02-01"

# Best of the 90-trial Optuna study in notebooks/Model_Training.ipynb.
# Used as-is unless --n-trials asks for a fresh search.
DEFAULT_PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.0481,
    "max_depth": 8,
    "num_leaves": 98,
}
FIXED_PARAMS = {"random_state": 42, "verbose": -1}


# ── Data access ──────────────────────────────────────────────────────────────
def fetch_features_snowflake() -> pd.DataFrame:
    import snowflake.connector
    from dotenv import load_dotenv

    load_dotenv()
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "compute_wh"),
        database=os.getenv("SNOWFLAKE_DATABASE", "nyc_taxi"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "surge_predictor"),
    )
    try:
        cs = conn.cursor()
        # dbt creates the table with a quoted lowercase name, hence the quotes.
        cs.execute('SELECT * FROM "fct_zone_features"')
        cols = [c[0] for c in cs.description]
        return pd.DataFrame(cs.fetchall(), columns=cols)
    finally:
        conn.close()


def fetch_features_duckdb(path: str) -> pd.DataFrame:
    import duckdb

    con = duckdb.connect(path, read_only=True)
    try:
        return con.execute("select * from surge_predictor.fct_zone_features").df()
    finally:
        con.close()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise names/types so the rest of the script is warehouse-agnostic."""
    df = df.rename(columns=str.lower)
    df["pickup_hour_ts"] = pd.to_datetime(df["pickup_hour_ts"])
    numeric = [c for c in df.columns if c not in ("pickup_hour_ts",)]
    df[numeric] = df[numeric].apply(pd.to_numeric)  # Snowflake returns Decimals
    return df.sort_values(["zone_id", "pickup_hour_ts"]).reset_index(drop=True)


# ── Modelling ────────────────────────────────────────────────────────────────
def fit_model(train: pd.DataFrame, columns: list[str], params: dict) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(**params, **FIXED_PARAMS)
    model.fit(train[columns], train[TARGET])
    return model


def tune(train: pd.DataFrame, val: pd.DataFrame, columns: list[str], n_trials: int) -> dict:
    """Optuna search scored on the validation slice (never on the holdout)."""
    import optuna

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        }
        model = fit_model(train, columns, params)
        return regression_metrics(val[TARGET], model.predict(val[columns]))["rmse"]

    study = optuna.create_study(direction="minimize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials)
    print(f"Optuna best validation RMSE: {study.best_value:.3f} with {study.best_params}")
    return study.best_params


def evaluate_horizon(horizon: int, train, val, holdout, params: dict) -> tuple[list[dict], pd.DataFrame, dict]:
    """Score baselines and both feature sets for one horizon.

    Returns the metric rows, the holdout predictions and the fitted models.
    """
    # Every method is scored on the same rows: those where all inputs exist.
    needed = feature_columns(horizon, "recent_plus_seasonal")
    val = val.dropna(subset=needed)
    holdout = holdout.dropna(subset=needed)
    train_rows = train.dropna(subset=feature_columns(horizon, "recent"))

    rows = []
    for name, col in baseline_columns(horizon).items():
        rows.append(
            {
                "horizon_h": horizon,
                "method": f"baseline: {name}",
                **{f"val_{k}": v for k, v in regression_metrics(val[TARGET], val[col]).items()},
                **{f"holdout_{k}": v for k, v in regression_metrics(holdout[TARGET], holdout[col]).items()},
            }
        )

    predictions = holdout[["zone_id", "pickup_hour_ts", TARGET]].copy()
    predictions["horizon_h"] = horizon
    models = {}
    for feature_set in FEATURE_SETS:
        columns = feature_columns(horizon, feature_set)
        model = fit_model(train_rows, columns, params)
        holdout_pred = model.predict(holdout[columns])
        rows.append(
            {
                "horizon_h": horizon,
                "method": f"lightgbm: {feature_set}",
                **{f"val_{k}": v for k, v in regression_metrics(val[TARGET], model.predict(val[columns])).items()},
                **{f"holdout_{k}": v for k, v in regression_metrics(holdout[TARGET], holdout_pred).items()},
            }
        )
        predictions[f"pred_{feature_set}"] = holdout_pred
        models[feature_set] = model
    return rows, predictions, models


def borough_breakdown(predictions: pd.DataFrame) -> pd.DataFrame:
    zones = pd.read_csv(ZONE_LOOKUP, keep_default_na=False)
    zones = zones.rename(columns={"LocationID": "zone_id", "Borough": "borough"})
    merged = predictions.merge(zones[["zone_id", "borough"]], on="zone_id", how="left")
    parts = []
    for horizon, part in merged.groupby("horizon_h"):
        by = metrics_by_group(part, "borough", TARGET, "pred_recent_plus_seasonal")
        by.insert(0, "horizon_h", horizon)
        parts.append(by)
    return pd.concat(parts, ignore_index=True)


# ── Reporting ────────────────────────────────────────────────────────────────
def write_reports(metrics: pd.DataFrame, boroughs: pd.DataFrame, meta: dict) -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    payload = {
        "meta": meta,
        "metrics": metrics.to_dict(orient="records"),
        "holdout_by_borough": boroughs.to_dict(orient="records"),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))

    lines = [
        "# Holdout results",
        "",
        f"Source: `{meta['source']}`. Train {meta['train_start']} to {meta['train_end']}, "
        f"validation to {meta['val_end']}, holdout {meta['holdout_start']} to {meta['holdout_end']} "
        f"({meta['holdout_rows']:,} zone-hours).",
        "",
        "| Horizon | Method | Holdout RMSE | Holdout MAE | Holdout WAPE |",
        "|---|---|---|---|---|",
    ]
    for r in metrics.itertuples():
        lines.append(
            f"| {r.horizon_h}h | {r.method} | {r.holdout_rmse:.2f} | {r.holdout_mae:.2f} | {r.holdout_wape:.1%} |"
        )
    lines += [
        "",
        "## LightGBM (recent + seasonal) by borough, holdout",
        "",
        "| Horizon | Borough | Zone-hours | Mean trips/zone-hour | RMSE | MAE | WAPE |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in boroughs.itertuples():
        # A percentage error is meaningless where there is almost no demand (e.g. Newark Airport).
        wape = "n/a" if pd.isna(r.wape) or r.mean_actual < 1 else f"{r.wape:.1%}"
        lines.append(
            f"| {r.horizon_h}h | {r.borough} | {r.rows:,} | {r.mean_actual:.1f} | {r.rmse:.2f} | {r.mae:.2f} | {wape} |"
        )
    (REPORTS_DIR / "holdout_metrics.md").write_text("\n".join(lines) + "\n")


def log_to_mlflow(metrics: pd.DataFrame, models: dict, params: dict, meta: dict) -> None:
    import mlflow
    import mlflow.lightgbm

    os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(EXPERIMENT_NAME)
    for r in metrics.to_dict(orient="records"):
        run_name = f"h{r['horizon_h']}_{r['method'].replace(': ', '_')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({"horizon_h": r["horizon_h"], "method": r["method"], **meta})
            mlflow.log_metrics({k: v for k, v in r.items() if k.startswith(("val_", "holdout_"))})
            if r["method"].startswith("lightgbm"):
                mlflow.log_params(params)
                feature_set = r["method"].split(": ")[1]
                mlflow.lightgbm.log_model(models[(r["horizon_h"], feature_set)], "model")


# ── Entry point ──────────────────────────────────────────────────────────────
def run(source: str, duckdb_path: str, holdout_start: str, n_trials: int, use_mlflow: bool) -> pd.DataFrame:
    raw = fetch_features_snowflake() if source == "snowflake" else fetch_features_duckdb(duckdb_path)
    df = prepare(raw)
    train, val, holdout = split_by_time(df, holdout_start)
    if holdout.empty:
        print(f"ERROR: no rows on or after holdout start {holdout_start}.", file=sys.stderr)
        sys.exit(1)
    print(f"Rows  train: {len(train):,}  val: {len(val):,}  holdout: {len(holdout):,}")

    params = dict(DEFAULT_PARAMS)
    if n_trials > 0:
        params = tune(train.dropna(subset=feature_columns(1)), val, feature_columns(1), n_trials)

    all_rows, all_predictions, all_models = [], [], {}
    for horizon in HORIZONS:
        rows, predictions, models = evaluate_horizon(horizon, train, val, holdout, params)
        all_rows += rows
        all_predictions.append(predictions)
        all_models.update({(horizon, name): m for name, m in models.items()})

    metrics = pd.DataFrame(all_rows)
    predictions = pd.concat(all_predictions, ignore_index=True)
    boroughs = borough_breakdown(predictions)
    meta = {
        "source": source,
        "train_start": train["pickup_hour_ts"].min(),
        "train_end": train["pickup_hour_ts"].max(),
        "val_end": val["pickup_hour_ts"].max(),
        "holdout_start": holdout["pickup_hour_ts"].min(),
        "holdout_end": holdout["pickup_hour_ts"].max(),
        "holdout_rows": int(len(holdout)),
        "n_trials": n_trials,
    }
    write_reports(metrics, boroughs, meta)
    summary_cols = ["horizon_h", "method", "val_rmse", "holdout_rmse", "holdout_mae", "holdout_wape"]
    print(metrics[summary_cols].to_string(index=False))

    if use_mlflow:
        log_to_mlflow(metrics, all_models, params, {k: str(v) for k, v in meta.items()})

    # Fail the task (and the DAG) if any score is not a real number.
    if metrics.filter(like="holdout_rmse").isna().any().any():
        print("ERROR: a holdout RMSE is NaN - failing task.", file=sys.stderr)
        sys.exit(1)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate 1-3 hour-ahead demand models.")
    parser.add_argument("--source", choices=["snowflake", "duckdb"], default="snowflake")
    parser.add_argument("--duckdb-path", default="nyc_taxi.duckdb")
    parser.add_argument("--holdout-start", default=DEFAULT_HOLDOUT_START,
                        help="First timestamp of the holdout period (default: 2026-02-01).")
    parser.add_argument("--n-trials", type=int, default=0,
                        help="Optuna trials. 0 (default) reuses the saved tuned parameters.")
    parser.add_argument("--no-mlflow", action="store_true", help="Skip MLflow logging (used in CI).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.source, args.duckdb_path, args.holdout_start, args.n_trials, not args.no_mlflow)
