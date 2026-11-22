import pytest
from fastapi.testclient import TestClient

from sqlforge.serve import executor


@pytest.fixture
def client(tpch_dir, monkeypatch):
    monkeypatch.setenv("DB_DIR", str(tpch_dir))
    monkeypatch.setattr(executor, "DBS", ("tpch",))
    with TestClient(executor.app) as c:
        yield c


def fake_model(sql: str):
    async def gen(db, question):
        return sql

    return gen


def test_ok_path(client, monkeypatch):
    monkeypatch.setattr(
        executor, "generate_sql", fake_model("SELECT n_name FROM nation ORDER BY n_name LIMIT 3")
    )
    r = client.post("/ask", json={"question": "first three nations", "db": "tpch"}).json()
    assert r["outcome"] == "ok" and r["row_count"] == 3 and r["columns"] == ["n_name"]


def test_guard_blocks_writes(client, monkeypatch):
    monkeypatch.setattr(executor, "generate_sql", fake_model("DELETE FROM nation"))
    assert client.post("/ask", json={"question": "delete all", "db": "tpch"}).json()["outcome"] == "unsafe"


def test_exec_error_and_bad_db(client, monkeypatch):
    monkeypatch.setattr(executor, "generate_sql", fake_model("SELECT nope FROM nation"))
    assert (
        client.post("/ask", json={"question": "bad column", "db": "tpch"}).json()["outcome"] == "exec_error"
    )
    assert client.post("/ask", json={"question": "x y z", "db": "nope"}).status_code == 400


def test_metrics_endpoint(client):
    assert "executor_requests_total" in client.get("/metrics").text


def test_consumer_parses_gateway_events():
    from sqlforge.serve.consumer import COLUMNS, parse_event

    raw = (
        b'{"ts":"2026-11-22T01:00:00.123Z","question":"q","db":"weather","cache":"hit_exact",'
        b'"status":200,"outcome":"ok","sql":"SELECT 1","latency_ms":1.5}'
    )
    row = parse_event(raw)
    assert len(row) == len(COLUMNS) and row[3] == "hit_exact" and row[0].year == 2026
    assert parse_event(b"not json") is None


def test_export_patches_rope_theta(tmp_path):
    import json

    pytest.importorskip("peft")
    from sqlforge.serve.export import patch_legacy_config

    p = tmp_path / "config.json"
    p.write_text(
        json.dumps({"dtype": "bfloat16", "rope_parameters": {"rope_theta": 1e6, "rope_type": "default"}})
    )
    patch_legacy_config(p)
    cfg = json.loads(p.read_text())
    assert cfg["rope_theta"] == 1e6 and cfg["torch_dtype"] == "bfloat16" and cfg["rope_scaling"] is None
