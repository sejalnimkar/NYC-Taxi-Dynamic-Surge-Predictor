-- The feature table must be a full zone x hour grid: every zone in the lookup
-- has a row for every hour. Fails (returns a row) if the row count is off.
with expected as (
    select
        (select count(*) from {{ ref('taxi_zone_lookup') }})
        * (select count(distinct pickup_hour_ts) from {{ ref('fct_zone_features') }}) as n_rows
),
actual as (
    select count(*) as n_rows from {{ ref('fct_zone_features') }}
)
select expected.n_rows as expected_rows, actual.n_rows as actual_rows
from expected, actual
where expected.n_rows <> actual.n_rows
