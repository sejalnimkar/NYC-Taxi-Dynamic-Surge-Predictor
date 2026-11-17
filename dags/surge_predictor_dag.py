"""
dags/surge_predictor_dag.py

Orchestrates the NYC Surge Predictor pipeline end to end:

    ingest_<month> -> load_<month>   (once per month, one after another)
        -> dbt seed -> dbt run -> dbt test
        -> train_model

Design notes:
- schedule=None: the dataset is a fixed set of months, so a recurring
  schedule would be theater. Trigger manually from the UI or CLI. To add a
  month, download its raw parquet into data/ and add it to MONTHS.
- Every step is safe to re-run: ingest overwrites its output, load replaces
  the month's rows in one transaction (see src/load.py), dbt rebuilds its
  tables, and train only writes new MLflow runs.
- Every task shells out via `docker exec` into the Jupyter container, which
  already has PySpark, the Snowflake connector, dbt and MLflow. This avoids
  installing dbt-snowflake into the Airflow image, which broke pip's
  dependency resolver earlier (dbt-core's pins conflict with Airflow's).
- SequentialExecutor + SQLite (see docker-compose.yml) means tasks run one
  at a time, which also keeps two Spark jobs from competing for memory.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

MONTHS = ["2026-01", "2026-02"]

JUPYTER_CONTAINER = "jupyter-notebook"
PYTHON = "/usr/bin/python3"
SRC_DIR = "/home/jovyan/work/src"
DBT_PROJECT_DIR = "/home/jovyan/work/nyc_surge_predictor"
DBT_PROFILES_DIR = "/home/jovyan/work/.dbt"

EXEC = f"docker exec {JUPYTER_CONTAINER}"
DBT_FLAGS = f"--project-dir {DBT_PROJECT_DIR} --profiles-dir {DBT_PROFILES_DIR}"

default_args = {
    "owner": "sejal",
    "retries": 0,
}

with DAG(
    dag_id="surge_predictor_pipeline",
    description="Ingest -> load -> dbt build/test -> train the NYC Surge Predictor models",
    default_args=default_args,
    schedule=None,  # manual trigger only; see design notes above
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nyc-taxi", "surge-predictor"],
) as dag:

    previous = None
    for month in MONTHS:
        task_suffix = month.replace("-", "_")

        ingest = BashOperator(
            task_id=f"ingest_{task_suffix}",
            bash_command=f"{EXEC} {PYTHON} {SRC_DIR}/ingest.py --month {month}",
            retries=1,
            retry_delay=timedelta(minutes=2),  # Spark is the flakiest step (OOM under tight Docker memory)
            doc_md=f"Clean {month} FHVHV trips with PySpark and write cleaned parquet to the shared data volume.",
        )

        load = BashOperator(
            task_id=f"load_{task_suffix}",
            bash_command=f"{EXEC} {PYTHON} {SRC_DIR}/load.py --month {month}",
            doc_md=f"Replace {month}'s rows in the raw Snowflake table with the cleaned parquet (idempotent).",
        )

        ingest >> load
        if previous is not None:
            previous >> ingest
        previous = load

    with TaskGroup(group_id="dbt_build") as dbt_build:

        dbt_seed = BashOperator(
            task_id="dbt_seed",
            bash_command=f"{EXEC} dbt seed {DBT_FLAGS}",
            doc_md="Load the zone lookup and holiday calendar seeds.",
        )

        dbt_run = BashOperator(
            task_id="dbt_run",
            bash_command=f"{EXEC} dbt run {DBT_FLAGS}",
            doc_md="Build staging -> hourly demand -> fct_zone_features in Snowflake.",
        )

        dbt_test = BashOperator(
            task_id="dbt_test",
            bash_command=f"{EXEC} dbt test {DBT_FLAGS}",
            doc_md="""
            Schema tests on staging plus grain, completeness, reconciliation
            and leakage checks on fct_zone_features. Training does not start
            unless these pass.
            """,
        )

        dbt_seed >> dbt_run >> dbt_test

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{EXEC} {PYTHON} {SRC_DIR}/train.py --source snowflake",
        doc_md="""
        Train one LightGBM model per horizon (1, 2, 3 hours ahead), score each
        against baselines on the held-out February data, and log to MLflow.
        """,
    )

    previous >> dbt_build >> train_model
