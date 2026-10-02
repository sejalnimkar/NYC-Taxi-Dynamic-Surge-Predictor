"""
src/ingest.py

Loads a month of raw FHVHV trip parquet data, cleans it, joins zone lookup
info, adds derived time columns, and writes the cleaned result back to
parquet.

Extracted from notebooks/ingest_data_and_EDA.ipynb — pipeline logic only.
EDA (schema prints, .show(), Folium maps, distribution checks) intentionally
left out; that stays in the notebook for exploration.

Usage:
    python ingest.py --month 2026-01
    python ingest.py --month 2026-02

Called by Airflow's `ingest` task via docker exec into the Spark/Jupyter
container, e.g.:
    docker exec jupyter-notebook python /home/jovyan/work/src/ingest.py --month 2026-01
"""

import argparse
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType


DATA_DIR = "/home/jovyan/work/data"
SPARK_MASTER = "spark://spark-master:7077"


def get_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("NYC_Taxi_Surge_Ingest")
        .master(SPARK_MASTER)
        .config("spark.executor.memory", "3g")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.cores", "1")
        # Smaller, more numerous shuffle partitions reduce peak memory per
        # task during the zone join — the stage that's been getting OOM-killed
        # (exit code 137) under constrained Docker memory.
        .config("spark.sql.shuffle.partitions", "16")
        .getOrCreate()
    )


def clean_trips(df):
    """Same row-level cleaning rules used for both Jan and Feb in the notebook."""
    return (
        df.dropDuplicates()
        .filter(F.col("trip_miles") > 0)
        .filter(F.col("trip_time") > 0)
        .filter(F.col("base_passenger_fare") > 0)
        .filter(F.col("driver_pay") > 0)
        .filter(F.col("pickup_datetime") < F.col("dropoff_datetime"))
        .filter(F.col("PULocationID").between(1, 265))
        .filter(F.col("DOLocationID").between(1, 265))
        .dropna(subset=["PULocationID", "DOLocationID", "pickup_datetime", "dropoff_datetime"])
    )


def join_zones(df_clean, zone_df):
    """Left-join pickup and dropoff zone lookup info onto the cleaned trips."""
    pu_zones = zone_df.select(
        F.col("LocationID").cast(IntegerType()).alias("PULocationID"),
        F.col("Zone").alias("PU_Zone"),
        F.col("Borough").alias("PU_Borough"),
    )
    do_zones = zone_df.select(
        F.col("LocationID").cast(IntegerType()).alias("DOLocationID"),
        F.col("Zone").alias("DO_Zone"),
        F.col("Borough").alias("DO_Borough"),
    )
    return (
        df_clean
        .join(pu_zones, on="PULocationID", how="left")
        .join(do_zones, on="DOLocationID", how="left")
    )


def add_derived_columns(df):
    return (
        df.withColumn("pickup_hour", F.hour("pickup_datetime"))
        .withColumn("day_of_week", F.dayofweek("pickup_datetime"))
        .withColumn("is_weekend", (F.dayofweek("pickup_datetime").isin([1, 7])).cast("integer"))
        .withColumn("pickup_date", F.to_date("pickup_datetime"))
    )


def run(month: str):
    """
    month: 'YYYY-MM', e.g. '2026-01'. Matches the raw file naming convention
    fhvhv_tripdata_{month}.parquet and controls the cleaned output filename.
    """
    input_path = f"{DATA_DIR}/fhvhv_tripdata_{month}.parquet"
    output_path = f"{DATA_DIR}/fhvhv_cleaned_{month}.parquet"
    zone_lookup_path = f"{DATA_DIR}/taxi_zone_lookup.csv"

    spark = get_spark_session()
    print(f"Spark version: {spark.version}")

    print(f"Reading raw trips from: {input_path}")
    df = spark.read.parquet(input_path)

    df_clean = clean_trips(df)
    # Note: no intermediate .count() here on purpose — each .count() forces
    # a full separate pass over the data, which was affordable running one
    # notebook at a time but adds real memory/compute pressure when the
    # whole stack (Spark + Jupyter + MLflow + Airflow) is running at once.

    zone_df = spark.read.csv(zone_lookup_path, header=True, inferSchema=True)
    df_with_zones = join_zones(df_clean, zone_df)
    df_final = add_derived_columns(df_with_zones)

    df_final.write.mode("overwrite").parquet(output_path)

    final_count = df_final.count()
    print(f"Saved to: {output_path}")
    print(f"Final row count: {final_count}")

    spark.stop()

    # Fail loudly if something upstream silently dropped everything —
    # useful as a DAG task failure signal rather than a false "success".
    if final_count == 0:
        print("ERROR: 0 rows in final output — failing task.", file=sys.stderr)
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Clean and join a month of FHVHV trip data.")
    parser.add_argument(
        "--month",
        required=True,
        help="Month to process, format YYYY-MM (e.g. 2026-01 or 2026-02).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.month)