"""
dags/surge_predictor_dag.py

Orchestrates the NYC Surge Predictor pipeline end to end:

    ingest (Jan + Feb)  ->  dbt run  ->  dbt test  ->  train_model

Design notes:
- schedule=None: the underlying dataset is fixed to Jan/Feb 2026, so a
  recurring schedule would be theater. Trigger manually from the UI or CLI.
- Every task shells out via `docker exec` into the sibling containers that
  already have the right environment (jupyter-notebook for PySpark/Snowflake/
  MLflow, and the dbt project's container for dbt itself). This avoids
  installing dbt-snowflake into the Airflow image, which broke pip's
  dependency resolver earlier (dbt-core's pins conflict with Airflow's).
- SequentialExecutor + SQLite (see docker-compose.yml) means tasks run one
  at a time regardless of the dependency graph — fine here since the graph
  is linear anyway.
"""

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup

# Container names as they appear in `docker compose ps` / docker-compose.yml.
# There's no separate dbt container — dbt is installed inside the Jupyter
# image and run from there, same as ingest.py/train.py.
JUPYTER_CONTAINER = "jupyter-notebook"
DBT_CONTAINER = JUPYTER_CONTAINER
DBT_PROJECT_DIR_IN_CONTAINER = "/home/jovyan/work/nyc_surge_predictor"
DBT_PROFILES_DIR = "/home/jovyan/work/.dbt"

SRC_DIR_IN_JUPYTER = "/home/jovyan/work/src"

default_args = {
    "owner": "sejal",
    "retries": 0,
}

with DAG(
    dag_id="surge_predictor_pipeline",
    description="Ingest -> dbt build/test -> train the NYC Surge Predictor model",
    default_args=default_args,
    schedule=None,  # manual trigger only; see design notes above
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["nyc-taxi", "surge-predictor"],
) as dag:

    ingest_task = BashOperator(
        task_id="ingest",
        bash_command=(
            f"docker exec {JUPYTER_CONTAINER} /usr/bin/python3 {SRC_DIR_IN_JUPYTER}/ingest.py --month 2026-01 && "
            f"docker exec {JUPYTER_CONTAINER} /usr/bin/python3 {SRC_DIR_IN_JUPYTER}/ingest.py --month 2026-02"
        ),
        retries=1,
        retry_delay=120,  # seconds; Spark jobs are the most failure-prone step per earlier debugging
        doc_md="""
        ### Ingest
        Runs `ingest.py` inside the Jupyter/Spark container for both January
        and February 2026 raw FHVHV trip data: cleans, joins taxi zone
        lookup info, adds derived time columns, and writes cleaned parquet
        back to the shared data volume. Retries once on failure since Spark
        jobs have been the flakiest step in this pipeline so far.
        """,
    )

    with TaskGroup(group_id="dbt_build") as dbt_build:

        dbt_run_task = BashOperator(
            task_id="dbt_run",
            bash_command=(
                f"docker exec {DBT_CONTAINER} "
                f"dbt run --project-dir {DBT_PROJECT_DIR_IN_CONTAINER} --profiles-dir {DBT_PROFILES_DIR}"
            ),
            doc_md="""
            ### dbt run
            Builds the staging -> intermediate -> fct_zone_features models
            in Snowflake from the cleaned trip data.
            """,
        )

        dbt_test_task = BashOperator(
            task_id="dbt_test",
            bash_command=(
                f"docker exec {DBT_CONTAINER} "
                f"dbt test --project-dir {DBT_PROJECT_DIR_IN_CONTAINER} --profiles-dir {DBT_PROFILES_DIR}"
            ),
            doc_md="""
            ### dbt test
            Runs the schema tests (not_null, accepted_values, relationships)
            validating fct_zone_features integrity before training.
            """,
        )

        dbt_run_task >> dbt_test_task

    train_model_task = BashOperator(
        task_id="train_model",
        bash_command=f"docker exec {JUPYTER_CONTAINER} /usr/bin/python3 {SRC_DIR_IN_JUPYTER}/train.py",
        doc_md="""
        ### Train model
        Pulls fct_zone_features from Snowflake, tunes a LightGBM regressor
        with Optuna, logs the run to MLflow, and evaluates the tuned model
        on the held-out February data.
        """,
    )

    ingest_task >> dbt_build >> train_model_task