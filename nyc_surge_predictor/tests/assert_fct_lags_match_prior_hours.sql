-- Leakage guard: pickup_lag_1h at hour T must equal pickup_count at T - 1h,
-- and the horizon-2 rolling window must not include hour T - 1h.
select cur.zone_id, cur.pickup_hour_ts
from {{ ref('fct_zone_features') }} cur
join {{ ref('fct_zone_features') }} prev
    on cur.zone_id = prev.zone_id
    and prev.pickup_hour_ts = {{ dbt.dateadd('hour', -1, 'cur.pickup_hour_ts') }}
where cur.pickup_lag_1h <> prev.pickup_count
   or cur.pickup_lag_2h <> prev.pickup_lag_1h
   or abs(cur.rolling_avg_3h_h2 - prev.rolling_avg_3h_h1) > 1e-6
