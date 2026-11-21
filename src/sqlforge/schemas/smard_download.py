"""Standalone SMARD downloader (fallback when Ruhrwatt's bronze layer is not on disk).

Writes the same hive layout Ruhrwatt uses:
  data/raw/smard/dataset=<name>/region=<region>/resolution=quarterhour/part-0.parquet
with columns (ts_utc TIMESTAMPTZ, value DOUBLE, is_native_15min BOOLEAN).

Filter ids are taken from Ruhrwatt (verified there against live files).
API: {BASE}/{filter}/{region}/index_quarterhour.json -> {"timestamps": [...]}
     {BASE}/{filter}/{region}/{filter}_{region}_quarterhour_{ts}.json -> {"series": [[ts_ms, v|null], ...]}
Data: Bundesnetzagentur | SMARD.de, CC BY 4.0.
"""

from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

BASE_URL = "https://www.smard.de/app/chart_data"
RESOLUTION = "quarterhour"
DAY_AHEAD_NATIVE_15MIN_FROM_MS = 1_759_269_600_000
TWO = ("DE", "Amprion")
DATASETS: dict[str, tuple[int, tuple[str, ...]]] = {
    "day_ahead_price": (4169, ("DE-LU",)),
    "grid_load": (410, TWO),
    "residual_load": (4359, TWO),
    "gen_lignite": (1223, TWO),
    "gen_wind_onshore": (4067, TWO),
    "gen_pv": (4068, TWO),
    "gen_hard_coal": (4069, TWO),
    "gen_gas": (4071, TWO),
    "gen_wind_offshore": (1225, ("DE",)),
    "gen_biomass": (4066, TWO),
    "gen_hydro": (1226, TWO),
}


def _get(session: requests.Session, url: str) -> dict:
    for attempt in range(5):
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code < 429 or r.status_code == 404:
            r.raise_for_status()
        time.sleep(2**attempt)
    r.raise_for_status()
    raise RuntimeError("unreachable")


def download(out: Path, weeks: int = 100, session: requests.Session | None = None) -> int:
    """Download the last `weeks` weekly files for every series. Returns number of rows written."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "sqlforge/0.1 (research)")
    total = 0
    for name, (fid, regions) in DATASETS.items():
        for region in regions:
            idx = _get(s, f"{BASE_URL}/{fid}/{region}/index_{RESOLUTION}.json")["timestamps"]
            ts_ms: list[int] = []
            vals: list[float | None] = []
            for wk in idx[-weeks:]:
                for t, v in _get(s, f"{BASE_URL}/{fid}/{region}/{fid}_{region}_{RESOLUTION}_{wk}.json")[
                    "series"
                ]:
                    if v is not None:
                        ts_ms.append(t)
                        vals.append(float(v))
            native = [name != "day_ahead_price" or t >= DAY_AHEAD_NATIVE_15MIN_FROM_MS for t in ts_ms]
            table = pa.table(
                {
                    "ts_utc": pa.array(ts_ms, pa.timestamp("ms", tz="UTC")),
                    "value": pa.array(vals, pa.float64()),
                    "is_native_15min": pa.array(native, pa.bool_()),
                }
            )
            d = out / f"dataset={name}" / f"region={region}" / f"resolution={RESOLUTION}"
            d.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, d / "part-0.parquet")
            total += table.num_rows
    return total
