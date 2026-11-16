# Local developer commands. None of these need Snowflake or Docker.
#   make install   install dev dependencies
#   make test      lint + unit tests (the Spark tests need Java 17+)
#   make demo      build a synthetic warehouse in DuckDB, run dbt (models +
#                  tests) on it, then train and score the models
DUCKDB_PATH ?= $(CURDIR)/nyc_taxi.duckdb
DBT = cd nyc_surge_predictor && DUCKDB_PATH=$(DUCKDB_PATH) dbt
DBT_FLAGS = --profiles-dir ../.dbt --target ci

.PHONY: install lint test fixture dbt-ci train-ci demo clean

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test: lint
	pytest -q

fixture:
	python scripts/make_fixture.py --db $(DUCKDB_PATH) --days 16

dbt-ci:
	$(DBT) build $(DBT_FLAGS)

train-ci:
	python src/train.py --source duckdb --duckdb-path $(DUCKDB_PATH) --holdout-start 2026-01-13 --no-mlflow

demo: fixture dbt-ci train-ci

clean:
	rm -f nyc_taxi.duckdb nyc_taxi.duckdb.wal
	rm -rf nyc_surge_predictor/target nyc_surge_predictor/logs
