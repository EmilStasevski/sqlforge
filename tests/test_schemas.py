import duckdb
import pytest

from sqlforge.schemas import build


@pytest.fixture
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "DB_DIR", tmp_path / "db")
    monkeypatch.setattr(build, "RAW_DIR", tmp_path / "raw")
    return tmp_path


def test_tpch_builds_eight_tables(in_tmp, monkeypatch):
    monkeypatch.setattr(build, "TPCH_SF", 0.01)
    counts = build.summary(build.build_tpch())
    assert set(counts) == {
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    }
    assert counts["nation"] == 25 and counts["region"] == 5


def test_weather_builds_from_cached_json(in_tmp, monkeypatch):
    def fake_fetch(lat, lon, cache):
        return {
            "elevation": 100.0,
            "daily": {"time": ["2023-01-01", "2023-01-02"], **{v: [1.0, None] for v in build.DAILY_VARS}},
            "hourly": {
                "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
                **{v: [2.0, 3.0] for v in build.HOURLY_VARS},
            },
        }

    monkeypatch.setattr(build, "_fetch_city", fake_fetch)
    path = build.build_weather()
    con = duckdb.connect(str(path), read_only=True)
    assert con.execute("SELECT COUNT(*) FROM city").fetchone()[0] == len(build.CITIES)
    assert con.execute("SELECT COUNT(*) FROM weather_daily").fetchone()[0] == 2 * len(build.CITIES)
    assert con.execute("SELECT typeof(date) FROM weather_daily LIMIT 1").fetchone()[0] == "DATE"
    assert (
        con.execute("SELECT COUNT(*) FROM weather_daily WHERE precipitation_sum IS NULL").fetchone()[0] == 6
    )
    assert con.execute("SELECT typeof(ts) FROM weather_hourly LIMIT 1").fetchone()[0] == "TIMESTAMP"


def test_city_names_keep_umlauts():
    names = {c[1] for c in build.CITIES}
    assert {"Köln", "Düsseldorf", "Münster"} <= names
