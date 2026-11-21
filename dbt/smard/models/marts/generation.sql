select
    s.hour_utc,
    s.region as region_code,
    replace(s.dataset, 'gen_', '') as source_code,
    round(s.value, 2) as generation_mwh
from {{ ref('stg_smard_hourly') }} s
where s.dataset like 'gen\_%' escape '\'
