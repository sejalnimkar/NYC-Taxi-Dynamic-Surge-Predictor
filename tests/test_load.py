"""The load step's SQL is built by pure functions so it can be checked without Snowflake."""

from datetime import date

import pytest

import load


def test_month_bounds():
    assert load.month_bounds("2026-01") == (date(2026, 1, 1), date(2026, 2, 1))
    assert load.month_bounds("2026-12") == (date(2026, 12, 1), date(2027, 1, 1))


@pytest.mark.parametrize("bad", ["2026-13", "2026-1", "26-01", "2026-01-01", "2026-01; DROP TABLE x"])
def test_bad_months_are_rejected_before_reaching_sql(bad):
    with pytest.raises(ValueError):
        load.month_bounds(bad)
    with pytest.raises(ValueError):
        load.stage_name(bad)


def test_delete_and_count_cover_exactly_one_month():
    for sql in (load.delete_statement("2026-02"), load.count_statement("2026-02")):
        assert "\"pickup_date\" >= '2026-02-01'" in sql
        assert "\"pickup_date\" < '2026-03-01'" in sql


def test_copy_forces_reload_from_the_months_own_stage():
    sql = load.copy_statement("2026-02")
    assert "@fhvhv_stage_2026_02" in sql and "FORCE = TRUE" in sql


def test_setup_stages_only_that_months_files():
    statements = load.setup_statements("2026-02", data_dir="/data")
    put = next(s for s in statements if s.startswith("PUT"))
    assert "file:///data/fhvhv_cleaned_2026-02.parquet/part-*.parquet" in put
    assert not any(s.lstrip().upper().startswith(("DELETE", "DROP", "TRUNCATE")) for s in statements)
