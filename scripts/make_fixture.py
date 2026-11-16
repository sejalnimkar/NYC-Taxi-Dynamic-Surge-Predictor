"""
scripts/make_fixture.py

Builds a small, synthetic copy of the raw trips table in a local DuckDB file
so the dbt project and the training script can run end to end without a
Snowflake account or the 900 MB of real TLC data.

The fixture has the same column names and types as the real raw table
(nyc_taxi.surge_predictor.fhvhv_trips), including timestamps stored as
integer microseconds, so the same dbt models run unchanged.

The data is random but has a daily demand cycle, so models have something to
learn. Numbers produced from it mean nothing; it exists to prove the code runs.

Usage:
    python scripts/make_fixture.py --db nyc_taxi.duckdb --days 16
"""

import argparse
from datetime import datetime

import duckdb
import numpy as np
import pandas as pd

SCHEMA = "surge_predictor"
TABLE = "fhvhv_trips"
START = datetime(2026, 1, 1)
ACTIVE_ZONES = [4, 13, 43, 48, 79, 87, 100, 132, 138, 161, 162, 230, 236, 237, 249]
EPOCH = datetime(1970, 1, 1)


def _micros(ts: pd.Series) -> pd.Series:
    return ((ts - EPOCH) // pd.Timedelta(microseconds=1)).astype("int64")


def build_trips(days: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = pd.date_range(START, periods=days * 24, freq="h")
    hour_of_day = hours.hour.to_numpy()
    # Two daily peaks (morning and evening) on top of a base level.
    daily_shape = (
        1.0
        + 2.0 * np.exp(-((hour_of_day - 8) ** 2) / 8)
        + 3.0 * np.exp(-((hour_of_day - 18) ** 2) / 10)
    )

    frames = []
    for i, zone in enumerate(ACTIVE_ZONES):
        zone_scale = 1.0 + (i % 5)  # some zones are busier than others
        counts = rng.poisson(daily_shape * zone_scale)
        pickup_hour = np.repeat(hours.to_numpy(), counts)
        n = len(pickup_hour)
        pickup = pd.Series(pickup_hour) + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
        trip_time = rng.integers(180, 3600, n)
        frames.append(
            pd.DataFrame(
                {
                    "pickup_ts": pickup,
                    "trip_time": trip_time,
                    "PULocationID": zone,
                    "DOLocationID": rng.choice(ACTIVE_ZONES, n),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    n = len(df)
    dropoff = df["pickup_ts"] + pd.to_timedelta(df["trip_time"], unit="s")
    fare = np.round(rng.uniform(8, 80, n), 2)

    return pd.DataFrame(
        {
            "hvfhs_license_num": rng.choice(["HV0003", "HV0005"], n, p=[0.73, 0.27]),
            "dispatching_base_num": "B03404",
            "originating_base_num": "B03404",
            "request_datetime": _micros(df["pickup_ts"] - pd.Timedelta(minutes=5)),
            "on_scene_datetime": _micros(df["pickup_ts"] - pd.Timedelta(minutes=1)),
            "pickup_datetime": _micros(df["pickup_ts"]),
            "dropoff_datetime": _micros(dropoff),
            "PULocationID": df["PULocationID"].astype("int32"),
            "DOLocationID": df["DOLocationID"].astype("int32"),
            "PU_Zone": "fixture",
            "PU_Borough": "fixture",
            "DO_Zone": "fixture",
            "DO_Borough": "fixture",
            "trip_miles": np.round(rng.uniform(0.5, 20, n), 2),
            "trip_time": df["trip_time"].astype("int64"),
            "base_passenger_fare": fare,
            "tolls": 0.0,
            "bcf": np.round(fare * 0.03, 2),
            "sales_tax": np.round(fare * 0.08875, 2),
            "congestion_surcharge": 2.75,
            "airport_fee": 0.0,
            "cbd_congestion_fee": 1.5,
            "tips": 0.0,
            "driver_pay": np.round(fare * 0.75, 2),
            "shared_request_flag": "N",
            "shared_match_flag": "N",
            "wav_request_flag": "N",
            "wav_match_flag": "N",
            "access_a_ride_flag": "N",
            "pickup_hour": df["pickup_ts"].dt.hour.astype("int32"),
            "day_of_week": (df["pickup_ts"].dt.dayofweek + 1) % 7 + 1,  # Spark: 1 = Sunday
            "is_weekend": (df["pickup_ts"].dt.dayofweek >= 5).astype("int32"),
            "pickup_date": df["pickup_ts"].dt.date,
        }
    )


def write_fixture(db_path: str, days: int) -> int:
    trips = build_trips(days)  # noqa: F841 - referenced by DuckDB via replacement scan
    con = duckdb.connect(db_path)
    con.execute(f"create schema if not exists {SCHEMA}")
    con.execute(f"create or replace table {SCHEMA}.{TABLE} as select * from trips")
    n = con.execute(f"select count(*) from {SCHEMA}.{TABLE}").fetchone()[0]
    con.close()
    return n


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a synthetic raw trips table in DuckDB.")
    parser.add_argument("--db", default="nyc_taxi.duckdb", help="DuckDB file to create or update.")
    parser.add_argument("--days", type=int, default=16, help="Days of synthetic trips to generate.")
    args = parser.parse_args()
    rows = write_fixture(args.db, args.days)
    print(f"Wrote {rows:,} synthetic trips to {args.db} ({SCHEMA}.{TABLE})")
