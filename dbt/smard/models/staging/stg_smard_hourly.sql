-- 15-min SMARD values -> complete UTC hours. Energy (MWh) is summed, prices are averaged.
select
    date_trunc('hour', timezone('UTC', ts_utc)) as hour_utc,
    dataset,
    region,
    case when dataset = 'day_ahead_price' then avg(value) else sum(value) end as value
from {{ source('raw', 'smard_raw') }}
where value is not null
group by 1, 2, 3
having count(*) = 4
