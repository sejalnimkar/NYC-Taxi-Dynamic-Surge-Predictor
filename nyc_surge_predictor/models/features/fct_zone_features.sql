{{ config(materialized='table') }}

{#
  One row per zone per hour, on a complete (zero-filled) zone x hour grid.

  The row for hour T holds the demand in hour T (the forecasting target) plus
  features describing the hours BEFORE T. To forecast T from h hours earlier,
  a model may only use what was known at T - h. The features are therefore
  grouped by forecast horizon:

    horizon h uses  pickup_lag_{h}h, pickup_lag_{h+1}h, pickup_lag_{h+2}h,
                    rolling_avg_3h_h{h}, rolling_avg_24h_h{h}

  Calendar columns describe hour T itself and are known in advance. The 24h
  and 168h lags (same hour yesterday / last week) are usable at every horizon.

  All windows are row-based. That is only correct because the grid has no
  missing hours, which tests/assert_fct_hours_are_contiguous.sql enforces.
#}

{% set horizons = [1, 2, 3] %}
{% set hourly_lags = [1, 2, 3, 4, 5] %}

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
),

holidays as (
    select holiday_date from {{ ref('holidays') }}
)

select
    j.zone_id,
    j.pickup_hour_ts,
    j.pickup_count,

    {% for n in hourly_lags -%}
    lag(j.pickup_count, {{ n }}) over (partition by j.zone_id order by j.pickup_hour_ts) as pickup_lag_{{ n }}h,
    {% endfor -%}
    lag(j.pickup_count, 24) over (partition by j.zone_id order by j.pickup_hour_ts) as pickup_lag_24h,
    lag(j.pickup_count, 168) over (partition by j.zone_id order by j.pickup_hour_ts) as pickup_lag_168h,

    {% for h in horizons -%}
    cast(avg(j.pickup_count) over (
        partition by j.zone_id order by j.pickup_hour_ts
        rows between {{ h + 2 }} preceding and {{ h }} preceding
    ) as float) as rolling_avg_3h_h{{ h }},
    cast(avg(j.pickup_count) over (
        partition by j.zone_id order by j.pickup_hour_ts
        rows between {{ h + 23 }} preceding and {{ h }} preceding
    ) as float) as rolling_avg_24h_h{{ h }},
    {% endfor -%}

    -- Same values as the _h1 columns, kept under their original names so the
    -- exploration notebooks keep working.
    cast(avg(j.pickup_count) over (
        partition by j.zone_id order by j.pickup_hour_ts
        rows between 3 preceding and 1 preceding
    ) as float) as rolling_avg_3h,
    cast(avg(j.pickup_count) over (
        partition by j.zone_id order by j.pickup_hour_ts
        rows between 24 preceding and 1 preceding
    ) as float) as rolling_avg_24h,

    dayofweek(j.pickup_hour_ts) as day_of_week,
    case when dayofweek(j.pickup_hour_ts) in (0, 6) then 1 else 0 end as is_weekend,
    hour(j.pickup_hour_ts) as hour_of_day,
    case when hd.holiday_date is not null then 1 else 0 end as is_holiday
from joined j
left join holidays hd
    on cast(j.pickup_hour_ts as date) = hd.holiday_date
