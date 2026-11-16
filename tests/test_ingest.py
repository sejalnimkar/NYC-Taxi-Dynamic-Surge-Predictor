"""Unit tests for the PySpark cleaning and enrichment steps in src/ingest.py."""

from datetime import datetime

import pytest

import ingest

pytestmark = pytest.mark.spark

COLUMNS = [
    "trip_miles", "trip_time", "base_passenger_fare", "driver_pay",
    "pickup_datetime", "dropoff_datetime", "PULocationID", "DOLocationID",
]
T0 = datetime(2026, 1, 3, 8, 15)   # a Saturday
T1 = datetime(2026, 1, 3, 8, 45)


def trip(**overrides):
    row = {
        "trip_miles": 2.5, "trip_time": 1800, "base_passenger_fare": 20.0, "driver_pay": 15.0,
        "pickup_datetime": T0, "dropoff_datetime": T1, "PULocationID": 100, "DOLocationID": 161,
    }
    row.update(overrides)
    return tuple(row[c] for c in COLUMNS)


def test_clean_trips_keeps_a_valid_trip(spark):
    df = spark.createDataFrame([trip()], COLUMNS)
    assert ingest.clean_trips(df).count() == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"trip_miles": 0.0},
        {"trip_time": 0},
        {"base_passenger_fare": -5.0},
        {"driver_pay": 0.0},
        {"dropoff_datetime": T0},                      # ends when it starts
        {"pickup_datetime": T1, "dropoff_datetime": T0},  # ends before it starts
        {"PULocationID": 0},
        {"PULocationID": 266},
        {"DOLocationID": 999},
    ],
)
def test_clean_trips_drops_invalid_trips(spark, bad):
    df = spark.createDataFrame([trip(), trip(**bad)], COLUMNS)
    cleaned = ingest.clean_trips(df)
    assert cleaned.count() == 1
    assert cleaned.first()["trip_miles"] == 2.5


def test_clean_trips_removes_exact_duplicates(spark):
    df = spark.createDataFrame([trip(), trip(), trip(trip_miles=3.0)], COLUMNS)
    assert ingest.clean_trips(df).count() == 2


def test_join_zones_adds_names_and_never_drops_trips(spark):
    trips = spark.createDataFrame([trip(), trip(PULocationID=5, trip_miles=9.0)], COLUMNS)
    zones = spark.createDataFrame(
        [(100, "Manhattan", "Garment District"), (161, "Manhattan", "Midtown Center")],
        ["LocationID", "Borough", "Zone"],
    )
    joined = ingest.join_zones(trips, zones)
    assert joined.count() == 2  # left join: the trip from unknown zone 5 survives
    known = joined.filter("PULocationID = 100").first()
    assert (known["PU_Zone"], known["PU_Borough"], known["DO_Zone"]) == (
        "Garment District", "Manhattan", "Midtown Center",
    )
    assert joined.filter("PULocationID = 5").first()["PU_Zone"] is None


def test_add_derived_columns(spark):
    monday = datetime(2026, 1, 5, 23, 59)
    df = spark.createDataFrame(
        [trip(), trip(pickup_datetime=monday, dropoff_datetime=datetime(2026, 1, 6, 0, 30))], COLUMNS
    )
    rows = {r["pickup_hour"]: r for r in ingest.add_derived_columns(df).collect()}
    saturday = rows[8]
    assert saturday["day_of_week"] == 7 and saturday["is_weekend"] == 1  # Spark: 1 = Sunday
    assert str(saturday["pickup_date"]) == "2026-01-03"
    assert rows[23]["day_of_week"] == 2 and rows[23]["is_weekend"] == 0
