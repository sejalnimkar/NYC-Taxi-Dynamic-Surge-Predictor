"""
scripts/load_hourly_counts.py

Loads a CSV of hourly pickup counts (zone_id, pickup_hour_ts, pickup_count)
into a local DuckDB file as int_hourly_zone_demand, so the feature model and
the training script can run on the real aggregates without Snowflake or the
40M raw trip rows.

The CSV is produced from the cleaned parquet by counting trips per zone per
hour, which is exactly what the dbt model int_hourly_zone_demand computes.

Usage:
    python scripts/load_hourly_counts.py --csv data/hourly_zone_demand.csv --db nyc_taxi.duckdb
    cd nyc_surge_predictor
    dbt seed --profiles-dir ../.dbt --target ci
    dbt run  --profiles-dir ../.dbt --target ci --select fct_zone_features
    dbt test --profiles-dir ../.dbt --target ci --select fct_zone_features --exclude assert_fct_reconciles_to_staging
"""

import argparse

import duckdb

SCHEMA = "surge_predictor"


def load(csv_path: str, db_path: str) -> tuple[int, int]:
    con = duckdb.connect(db_path)
    con.execute(f"create schema if not exists {SCHEMA}")
    con.execute(
        f"""
        create or replace table {SCHEMA}.int_hourly_zone_demand as
        select
            cast(zone_id as integer) as zone_id,
            cast(pickup_hour_ts as timestamp) as pickup_hour_ts,
            cast(pickup_count as bigint) as pickup_count
        from read_csv(?, header = true)
        """,
        [csv_path],
    )
    rows, trips = con.execute(f"select count(*), sum(pickup_count) from {SCHEMA}.int_hourly_zone_demand").fetchone()
    con.close()
    return rows, trips


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load hourly pickup counts into DuckDB.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--db", default="nyc_taxi.duckdb")
    args = parser.parse_args()
    n_rows, n_trips = load(args.csv, args.db)
    print(f"Loaded {n_rows:,} zone-hours covering {n_trips:,} trips into {args.db}")
