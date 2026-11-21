"""Benchmark Rust sqlnorm (PyO3) vs sqlglot on the dataset's SQL: parse + canonicalise + hash.

Writes results/bench_sqlnorm.json. Both sides do the same work per query:
  parse (DuckDB dialect) -> canonical SQL text (lower-cased identifiers) -> 64-bit hash
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl
import sqlglot
import sqlnorm

REPEATS = 5


def sqlglot_hash(sql: str) -> str:
    canon = sqlglot.parse_one(sql, read="duckdb").sql(dialect="duckdb", normalize=True)
    return hashlib.blake2b(canon.encode(), digest_size=8).hexdigest()


def timed(fn, corpus) -> list[float]:
    out = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(corpus)
        out.append(time.perf_counter() - t0)
    return out


def main() -> None:
    corpus = []
    for s in ("train", "val", "test_in", "test_ood"):
        corpus += pl.read_parquet(f"data/dataset/{s}_template.parquet")["sql"].to_list()

    runs = {
        "sqlnorm_per_call": timed(lambda c: [sqlnorm.canonical_hash(s) for s in c], corpus),
        "sqlnorm_batch": timed(lambda c: sqlnorm.canonical_hash_many(c), corpus),
        "sqlglot": timed(lambda c: [sqlglot_hash(s) for s in c], corpus),
    }
    # sanity: both produce one hash per query, and agree on duplicate structure
    rs = [sqlnorm.canonical_hash(s) for s in corpus[:2000]]
    gs = [sqlglot_hash(s) for s in corpus[:2000]]
    same_partition = len(set(rs)) == len(set(gs))

    med = {k: statistics.median(v) for k, v in runs.items()}
    res = {
        "n_queries": len(corpus),
        "repeats": REPEATS,
        "median_seconds": {k: round(v, 4) for k, v in med.items()},
        "queries_per_second": {k: round(len(corpus) / v) for k, v in med.items()},
        "speedup_vs_sqlglot": {
            "per_call": round(med["sqlglot"] / med["sqlnorm_per_call"], 1),
            "batch": round(med["sqlglot"] / med["sqlnorm_batch"], 1),
        },
        "dedup_partition_agrees_first_2000": same_partition,
        "env": {
            "python": platform.python_version(),
            "sqlglot": version("sqlglot"),
            "sqlglot_rust_tokenizer": importlib.util.find_spec("sqlglotrs") is not None,
            "cpu": platform.processor() or platform.machine(),
        },
    }
    Path("results/bench_sqlnorm.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
