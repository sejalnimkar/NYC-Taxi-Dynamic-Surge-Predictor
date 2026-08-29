with source as (
    select * from {{ source('raw', 'fhvhv_trips') }}
),

cleaned as (
    select
        "hvfhs_license_num"::varchar                       as hvfhs_license_num,
        "dispatching_base_num"::varchar                     as dispatching_base_num,
        "originating_base_num"::varchar                     as originating_base_num,
        to_timestamp_ntz("pickup_datetime", 6)              as pickup_datetime,
        to_timestamp_ntz("dropoff_datetime", 6)             as dropoff_datetime,
        to_timestamp_ntz("request_datetime", 6)             as request_datetime,
        to_timestamp_ntz("on_scene_datetime", 6)            as on_scene_datetime,
        "PULocationID"::integer                             as pickup_location_id,
        "DOLocationID"::integer                             as dropoff_location_id,
        "PU_Zone"::varchar                                  as pickup_zone,
        "PU_Borough"::varchar                                as pickup_borough,
        "DO_Zone"::varchar                                  as dropoff_zone,
        "DO_Borough"::varchar                                as dropoff_borough,
        "trip_miles"::float                                 as trip_miles,
        "trip_time"::integer                                as trip_time_seconds,
        "base_passenger_fare"::float                        as base_passenger_fare,
        "tolls"::float                                      as tolls,
        "bcf"::float                                         as bcf,
        "sales_tax"::float                                  as sales_tax,
        "congestion_surcharge"::float                       as congestion_surcharge,
        "airport_fee"::float                                as airport_fee,
        "cbd_congestion_fee"::float                          as cbd_congestion_fee,
        "tips"::float                                       as tips,
        "driver_pay"::float                                 as driver_pay,
        "shared_request_flag"::varchar                      as shared_request_flag,
        "shared_match_flag"::varchar                        as shared_match_flag,
        "wav_request_flag"::varchar                         as wav_request_flag,
        "wav_match_flag"::varchar                           as wav_match_flag,
        "access_a_ride_flag"::varchar                       as access_a_ride_flag,
        "pickup_hour"::integer                              as pickup_hour,
        "day_of_week"::integer                              as day_of_week,
        "is_weekend"::integer                                as is_weekend,
        "pickup_date"::date                                  as pickup_date
    from source
    where "PULocationID" is not null
      and "DOLocationID" is not null
      and "pickup_datetime" is not null
      and "trip_miles" >= 0
      and "trip_time" >= 0
)

select * from cleaned
