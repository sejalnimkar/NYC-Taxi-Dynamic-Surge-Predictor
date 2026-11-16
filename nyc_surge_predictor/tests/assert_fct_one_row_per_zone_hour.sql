-- Grain check: (zone_id, pickup_hour_ts) must be unique. Returns offending keys.
select zone_id, pickup_hour_ts, count(*) as n_rows
from {{ ref('fct_zone_features') }}
group by 1, 2
having count(*) > 1
