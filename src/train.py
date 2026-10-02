"""
src/train.py

Pulls engineered features from Snowflake (fct_zone_features, built by dbt),
tunes a LightGBM regressor with Optuna, logs runs to MLflow, and evaluates
the tuned model on the held-out February data.

Extracted from notebooks/Model_Training.ipynb — the winning model path only
(LightGBM, tuned). The GBDT/XGBoost/naive baselines that were compared
during exploration are intentionally left out of the production script;
they still live in the notebook for anyone who wants to see the comparison.

Usage:
    python train.py
    python train.py --n-trials 20

Called by Airflow's `train_model` task via docker exec, e.g.:
    docker exec jupyter-notebook python /home/jovyan/work/src/train.py
"""

import argparse
import os
import sys

import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
import snowflake.connector
from dotenv import load_dotenv
from optuna.integration.mlflow import MLflowCallback
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "nyc_surge_gbdt"

FEATURES = [
    "PICKUP_LAG_1H", "PICKUP_LAG_2H", "PICKUP_LAG_3H",
    "ROLLING_AVG_3H", "ROLLING_AVG_24H",
    "DAY_OF_WEEK", "IS_WEEKEND", "HOUR_OF_DAY", "IS_HOLIDAY",
]
TARGET = "PICKUP_COUNT"

FEB_SPLIT = "2026-02-01"


def get_snowflake_cursor():
    load_dotenv()
    conn = snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse="compute_wh",
        database="nyc_taxi",
        schema="surge_predictor",
    )
    return conn.cursor()


def fetch_features(cs, where_clause: str = "") -> pd.DataFrame:
    query = f'SELECT * FROM "fct_zone_features" {where_clause} ORDER BY ZONE_ID, PICKUP_HOUR_TS;'
    cs.execute(query)
    cols = [c[0] for c in cs.description]
    df = pd.DataFrame(cs.fetchall(), columns=cols)
    for col in ["ROLLING_AVG_3H", "ROLLING_AVG_24H"]:
        df[col] = df[col].astype(float)
    return df


def time_split(df_jan: pd.DataFrame):
    """80/20 time-based split within January; Feb is never seen here."""
    split_time = df_jan["PICKUP_HOUR_TS"].quantile(0.8)
    train = df_jan[df_jan["PICKUP_HOUR_TS"] <= split_time]
    val = df_jan[df_jan["PICKUP_HOUR_TS"] > split_time]
    return train, val


def tune(X_train, y_train, X_val, y_val, n_trials: int) -> dict:
    mlflc = MLflowCallback(tracking_uri=MLFLOW_TRACKING_URI, metric_name="rmse")

    @mlflc.track_in_mlflow()
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "random_state": 42,
            "verbose": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    study = optuna.create_study(direction="minimize", study_name="lightgbm_tuning")
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflc])

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)
    return study.best_params


def train_final_model(X_train, y_train, X_val, y_val, best_params: dict):
    final_params = {**best_params, "random_state": 42, "verbose": -1}

    with mlflow.start_run(run_name="lightgbm_tuned_final"):
        model = lgb.LGBMRegressor(**final_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

        mlflow.log_params({"model": "lightgbm_tuned", **final_params})
        mlflow.log_metric("rmse", rmse)
        mlflow.lightgbm.log_model(model, "model")

        print("Final tuned RMSE (Jan validation):", rmse)

    return model


def evaluate_on_february(model, cs) -> float:
    df_feb_eval = fetch_features(cs, where_clause=f"WHERE PICKUP_HOUR_TS >= '{FEB_SPLIT}'")
    print(f"Feb holdout shape: {df_feb_eval.shape}")

    X_feb = df_feb_eval[FEATURES]
    y_feb = df_feb_eval[TARGET]

    feb_preds = model.predict(X_feb)
    feb_rmse = np.sqrt(mean_squared_error(y_feb, feb_preds))
    print("Feb (out-of-sample) RMSE:", feb_rmse)

    with mlflow.start_run(run_name="lightgbm_tuned_feb_eval"):
        mlflow.log_param("eval_set", "february_holdout")
        mlflow.log_metric("rmse", feb_rmse)

    return feb_rmse


def run(n_trials: int):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    cs = get_snowflake_cursor()

    df = fetch_features(cs)
    df_jan = df[df["PICKUP_HOUR_TS"] < FEB_SPLIT].copy()
    print(f"Jan rows: {df_jan.shape}")

    train, val = time_split(df_jan)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = val[FEATURES], val[TARGET]

    best_params = tune(X_train, y_train, X_val, y_val, n_trials=n_trials)
    model = train_final_model(X_train, y_train, X_val, y_val, best_params)
    feb_rmse = evaluate_on_february(model, cs)

    # Fail the task if the model is clearly broken, rather than silently
    # logging a bad run as if it were fine.
    if feb_rmse is None or feb_rmse != feb_rmse:  # NaN check without numpy import here
        print("ERROR: Feb evaluation RMSE is invalid — failing task.", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Tune, train, and evaluate the surge-prediction model.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna trials for hyperparameter tuning (default: 30).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.n_trials)
