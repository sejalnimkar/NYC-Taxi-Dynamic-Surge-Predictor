-- Reconciliation: zero-filling and feature joins must not create or lose
-- trips. Total pickups in the feature table must equal staged trip rows.
with fct as (
    select sum(pickup_count) as n_trips from {{ ref('fct_zone_features') }}
),
stg as (
    select count(*) as n_trips from {{ ref('stg_fhvhv_trips') }}
)
select fct.n_trips as fct_trips, stg.n_trips as stg_trips
from fct, stg
where fct.n_trips <> stg.n_trips
