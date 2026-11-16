-- Lag and rolling features use row offsets ("3 rows back" = "3 hours back").
-- That only holds if no hour is missing from the grid. Fails if the number of
-- distinct hours differs from the span between the first and last hour.
with bounds as (
    select
        min(pickup_hour_ts) as first_hour,
        max(pickup_hour_ts) as last_hour,
        count(distinct pickup_hour_ts) as n_hours
    from {{ ref('fct_zone_features') }}
)
select *
from bounds
where {{ dbt.datediff('first_hour', 'last_hour', 'hour') }} + 1 <> n_hours
