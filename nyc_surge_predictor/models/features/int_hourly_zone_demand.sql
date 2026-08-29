{{ config(materialized='view') }}

select
    pickup_location_id as zone_id,
    date_trunc('hour', pickup_datetime) as pickup_hour_ts,
    count(*) as pickup_count
from {{ ref('stg_fhvhv_trips') }}
group by 1, 2