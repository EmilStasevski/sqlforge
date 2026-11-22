"""FastAPI executor: question -> model server (OpenAI-compatible chat API) -> Rust guard -> read-only DuckDB.

Env:
  MODEL_URL   OpenAI-compatible base URL (Ollama: http://host:11434/v1, llama.cpp server: http://host:8081/v1)
  MODEL_NAME  model id on that server (default: sqlforge)
  DB_DIR      directory with <db>.duckdb files (default: data/db)

Run: uv run uvicorn sqlforge.serve.executor:app --port 8000
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
import httpx
import sqlnorm
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field
from starlette.responses import Response

from sqlforge.eval.score import extract_sql
from sqlforge.prompt import SYSTEM_PROMPT
from sqlforge.schemas import card as card_mod
from sqlforge.schemas.variant import full_variant
from sqlforge.verify.execute import TIMEOUT_S, QueryTimeout

DBS = ("tpch", "smard", "weather")
MAX_RETURN_ROWS = 200

REQS = Counter("executor_requests_total", "Requests by outcome", ["db", "outcome"])
LAT = Histogram(
    "executor_stage_seconds",
    "Latency per stage",
    ["stage"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16),
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    db: str = "weather"


class AskResponse(BaseModel):
    sql: str
    columns: list[str] = []
    rows: list[list] = []
    row_count: int = 0
    outcome: str  # ok | unsafe | syntax | exec_error | timeout | model_error
    error: str | None = None
    timings_ms: dict[str, float] = {}


class State:
    ddl: dict[str, str] = {}
    db_paths: dict[str, Path] = {}
    client: httpx.AsyncClient | None = None


S = State()


@asynccontextmanager
async def lifespan(_: FastAPI):
    card_mod.DB_DIR = Path(os.environ.get("DB_DIR", "data/db"))
    for db in DBS:
        c = card_mod.load_card(db)
        S.ddl[db] = full_variant(c).ddl()
        S.db_paths[db] = c.db_path
    S.client = httpx.AsyncClient(
        base_url=os.environ.get("MODEL_URL", "http://localhost:11434/v1"), timeout=60
    )
    yield
    await S.client.aclose()


app = FastAPI(title="sqlforge executor", lifespan=lifespan)


async def generate_sql(db: str, question: str) -> str:
    body = {
        "model": os.environ.get("MODEL_NAME", "sqlforge"),
        "temperature": 0,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{S.ddl[db]}\n\nQuestion: {question}"},
        ],
    }
    r = await S.client.post("/chat/completions", json=body)
    r.raise_for_status()
    return extract_sql(r.json()["choices"][0]["message"]["content"])


def execute(db: str, sql: str, timeout_s: float = TIMEOUT_S) -> tuple[list[str], list[tuple]]:
    con = duckdb.connect(str(S.db_paths[db]), read_only=True)
    timer = threading.Timer(timeout_s, con.interrupt)
    try:
        con.execute("SET threads = 2")
        timer.start()
        cur = con.execute(sql)
        return [d[0] for d in cur.description], cur.fetchmany(MAX_RETURN_ROWS)
    except duckdb.InterruptException as e:
        raise QueryTimeout(sql) from e
    finally:
        timer.cancel()
        con.close()


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    if req.db not in DBS:
        raise HTTPException(400, f"db must be one of {DBS}")
    t: dict[str, float] = {}
    t0 = time.perf_counter()
    try:
        sql = await generate_sql(req.db, req.question)
    except (httpx.HTTPError, KeyError) as e:
        REQS.labels(req.db, "model_error").inc()
        return AskResponse(sql="", outcome="model_error", error=str(e)[:300])
    t["generate"] = (time.perf_counter() - t0) * 1000
    LAT.labels("generate").observe(t["generate"] / 1000)

    t1 = time.perf_counter()
    try:
        sqlnorm.normalize(sql)
        safe = sqlnorm.is_safe_select(sql)
    except ValueError:
        REQS.labels(req.db, "syntax").inc()
        return AskResponse(sql=sql, outcome="syntax", timings_ms=t)
    t["guard"] = (time.perf_counter() - t1) * 1000
    if not safe:
        REQS.labels(req.db, "unsafe").inc()
        return AskResponse(sql=sql, outcome="unsafe", timings_ms=t)

    t2 = time.perf_counter()
    try:
        cols, rows = await run_in_threadpool(execute, req.db, sql)
    except QueryTimeout:
        REQS.labels(req.db, "timeout").inc()
        return AskResponse(sql=sql, outcome="timeout", timings_ms=t)
    except duckdb.Error as e:
        REQS.labels(req.db, "exec_error").inc()
        return AskResponse(sql=sql, outcome="exec_error", error=str(e)[:300], timings_ms=t)
    t["execute"] = (time.perf_counter() - t2) * 1000
    LAT.labels("execute").observe(t["execute"] / 1000)
    REQS.labels(req.db, "ok").inc()
    json_rows = [
        [v if isinstance(v, (int, float, str, bool)) or v is None else str(v) for v in r] for r in rows
    ]
    return AskResponse(sql=sql, columns=cols, rows=json_rows, row_count=len(rows), outcome="ok", timings_ms=t)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "dbs": list(S.ddl)}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
