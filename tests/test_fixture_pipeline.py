"""The synthetic fixture must look like the real raw table, or CI proves nothing."""

import make_fixture


def test_fixture_matches_raw_table_contract():
    trips = make_fixture.build_trips(days=2)
    for col in ["pickup_datetime", "dropoff_datetime", "request_datetime", "on_scene_datetime"]:
        assert str(trips[col].dtype) == "int64"  # microseconds, as in Snowflake
    assert trips["PULocationID"].between(1, 265).all()
    assert (trips["dropoff_datetime"] > trips["pickup_datetime"]).all()
    assert set(trips["hvfhs_license_num"]) <= {"HV0003", "HV0005"}


def test_fixture_is_deterministic_and_has_no_empty_hours():
    a, b = make_fixture.build_trips(days=2), make_fixture.build_trips(days=2)
    assert a.equals(b)
    hours = (a["pickup_datetime"] // 3_600_000_000).nunique()
    assert hours == 48
