"""
src/load.py

Loads one month of cleaned trip parquet (the output of ingest.py) into the
raw Snowflake table nyc_taxi.surge_predictor.fhvhv_trips.

This step used to be run by hand from notebooks/Checks/Staging.ipynb. It is
now a script so the Airflow DAG covers the whole path from files to features.

The load is idempotent: running it twice for the same month leaves the same
rows as running it once. It does that by deleting the month's rows and
re-copying them inside a single transaction, so a failure part-way through
rolls back and never leaves the month half-loaded or doubled.

Usage:
    python load.py --month 2026-01

Called by Airflow's `load_<month>` task via docker exec.
"""

import argparse
import os
import re
import sys
from datetime import date

DATA_DIR = os.getenv("DATA_DIR", "/home/jovyan/work/data")
TABLE = "fhvhv_trips"
FILE_FORMAT = "my_parquet_format"


def month_bounds(month: str) -> tuple[date, date]:
    """'2026-01' -> (date(2026, 1, 1), date(2026, 2, 1)). The end is exclusive."""
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"month must look like YYYY-MM, got {month!r}")
    year, mon = int(month[:4]), int(month[5:])
    start = date(year, mon, 1)
    end = date(year + 1, 1, 1) if mon == 12 else date(year, mon + 1, 1)
    return start, end


def stage_name(month: str) -> str:
    month_bounds(month)  # validates the format before it is used in SQL
    return f"fhvhv_stage_{month.replace('-', '_')}"


def setup_statements(month: str, data_dir: str = DATA_DIR) -> list[str]:
    """Statements that stage the files. Safe to re-run; they touch no table rows."""
    stage = stage_name(month)
    return [
        f"CREATE FILE FORMAT IF NOT EXISTS {FILE_FORMAT} TYPE = PARQUET",
        # OR REPLACE empties the stage, so files from an earlier run cannot linger.
        f"CREATE OR REPLACE STAGE {stage}",
        f"PUT file://{data_dir}/fhvhv_cleaned_{month}.parquet/part-*.parquet @{stage} AUTO_COMPRESS=FALSE",
        # First-ever load: create the table from the parquet schema.
        f"""CREATE TABLE IF NOT EXISTS {TABLE} USING TEMPLATE (
            SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
            FROM TABLE(INFER_SCHEMA(LOCATION => '@{stage}', FILE_FORMAT => '{FILE_FORMAT}'))
        )""",
    ]


def delete_statement(month: str) -> str:
    start, end = month_bounds(month)
    return f"""DELETE FROM {TABLE} WHERE "pickup_date" >= '{start}' AND "pickup_date" < '{end}'"""


def copy_statement(month: str) -> str:
    # FORCE = TRUE: Snowflake skips files it has loaded before; after the DELETE
    # above we do want them loaded again.
    return f"""COPY INTO {TABLE} FROM @{stage_name(month)}
        FILE_FORMAT = (FORMAT_NAME = '{FILE_FORMAT}')
        MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE
        FORCE = TRUE"""


def count_statement(month: str) -> str:
    start, end = month_bounds(month)
    return f"""SELECT COUNT(*) FROM {TABLE} WHERE "pickup_date" >= '{start}' AND "pickup_date" < '{end}'"""


def connect():
    import snowflake.connector
    from dotenv import load_dotenv

    load_dotenv()
    return snowflake.connector.connect(
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        user=os.getenv("SNOWFLAKE_USER"),
        password=os.getenv("SNOWFLAKE_PASSWORD"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE", "compute_wh"),
        database=os.getenv("SNOWFLAKE_DATABASE", "nyc_taxi"),
        schema=os.getenv("SNOWFLAKE_SCHEMA", "surge_predictor"),
    )


def run(month: str) -> int:
    conn = connect()
    cs = conn.cursor()
    try:
        for statement in setup_statements(month):
            cs.execute(statement)

        cs.execute("BEGIN")
        try:
            cs.execute(delete_statement(month))
            deleted = cs.rowcount
            cs.execute(copy_statement(month))
            # COPY returns one row per file; column 3 is rows_loaded.
            copied = sum(row[3] for row in cs.fetchall() if len(row) > 3)
            cs.execute(count_statement(month))
            in_table = cs.fetchone()[0]
            if copied == 0 or in_table != copied:
                raise RuntimeError(
                    f"{month}: copied {copied:,} rows but the table holds {in_table:,} for that month. "
                    "The files may contain pickups dated outside the month."
                )
            cs.execute("COMMIT")
        except Exception:
            cs.execute("ROLLBACK")
            raise
        print(f"{month}: replaced {deleted:,} existing rows with {copied:,} rows.")
        return copied
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Idempotently load one month of cleaned trips into Snowflake.")
    parser.add_argument("--month", required=True, help="Month to load, format YYYY-MM.")
    args = parser.parse_args()
    try:
        run(args.month)
    except Exception as exc:  # non-zero exit so the Airflow task fails
        print(f"ERROR: load failed: {exc}", file=sys.stderr)
        sys.exit(1)
