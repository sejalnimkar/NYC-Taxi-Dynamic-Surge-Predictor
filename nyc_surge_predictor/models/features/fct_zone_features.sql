{{ config(materialized='table') }}

with hourly as (
    select * from {{ ref('int_hourly_zone_demand') }}
),

spined as (
    select
        z.locationid as zone_id,
        h.hour_ts as pickup_hour_ts
    from {{ ref('taxi_zone_lookup') }} z
    cross join (
        select distinct pickup_hour_ts as hour_ts from hourly
    ) h
),

joined as (
    select
        s.zone_id,
        s.pickup_hour_ts,
        coalesce(h.pickup_count, 0) as pickup_count
    from spined s
    left join hourly h
        on s.zone_id = h.zone_id
        and s.pickup_hour_ts = h.pickup_hour_ts
)

select
    zone_id,
    pickup_hour_ts,
    pickup_count,
    lag(pickup_count, 1) over (partition by zone_id order by pickup_hour_ts) as pickup_lag_1h,
    lag(pickup_count, 2) over (partition by zone_id order by pickup_hour_ts) as pickup_lag_2h,
    lag(pickup_count, 3) over (partition by zone_id order by pickup_hour_ts) as pickup_lag_3h,
    avg(pickup_count) over (
        partition by zone_id order by pickup_hour_ts
        rows between 3 preceding and 1 preceding
    ) as rolling_avg_3h,
    avg(pickup_count) over (
        partition by zone_id order by pickup_hour_ts
        rows between 24 preceding and 1 preceding
    ) as rolling_avg_24h,
    dayofweek(pickup_hour_ts) as day_of_week,
    case when dayofweek(pickup_hour_ts) in (0, 6) then 1 else 0 end as is_weekend,
    hour(pickup_hour_ts) as hour_of_day,
    case
        when date(pickup_hour_ts) in ('2026-01-01', '2026-01-19') then 1
        else 0
    end as is_holiday
from joined
