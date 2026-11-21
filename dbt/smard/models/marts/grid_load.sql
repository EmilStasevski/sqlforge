select
    hour_utc,
    region as region_code,
    round(max(case when dataset = 'grid_load' then value end), 2) as load_mwh,
    round(max(case when dataset = 'residual_load' then value end), 2) as residual_load_mwh
from {{ ref('stg_smard_hourly') }}
where dataset in ('grid_load', 'residual_load')
group by 1, 2
having load_mwh is not null and residual_load_mwh is not null
