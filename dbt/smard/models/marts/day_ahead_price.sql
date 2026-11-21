select
    hour_utc,
    region as region_code,
    round(value, 2) as price_eur_mwh
from {{ ref('stg_smard_hourly') }}
where dataset = 'day_ahead_price'
