"""Load test the gateway with vegeta: same request stream with the Redis cache disabled vs enabled.

Stream: N distinct test_ood questions, each sent REPEAT times in a seeded shuffled order; 30 % of repeats are
case/whitespace variants (exercise the normalized-question cache). Writes results/loadtest.json.

Requires the compose stack (`make up`) and vegeta on PATH (go install github.com/tsenart/vegeta/v12@v12.12.0).
Usage: uv run python scripts/loadtest.py [--n 120] [--repeat 5] [--rate 2]
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import polars as pl

GATEWAY = "http://localhost:8080"
SCRATCH = Path("data/loadtest")


def variant(q: str, rng: random.Random) -> str:
    return rng.choice([q.lower(), q.upper(), "  " + q.replace(" ", "  ") + " ", q.rstrip("?.") + " ?"])


def build_targets(n: int, repeat: int, seed: int = 0) -> list[dict]:
    df = pl.read_parquet("data/dataset/test_ood.parquet").sample(n=n, seed=seed)
    rng = random.Random(seed)
    stream = []
    for q in df["question"]:
        stream.append(q)
        stream += [variant(q, rng) if rng.random() < 0.3 else q for _ in range(repeat - 1)]
    rng.shuffle(stream)
    return [
        {
            "method": "POST",
            "url": f"{GATEWAY}/ask",
            "header": {"Content-Type": ["application/json"]},
            "body": base64.b64encode(json.dumps({"question": q, "db": "weather"}).encode()).decode(),
        }
        for q in stream
    ]


def metrics() -> dict[str, float]:
    txt = urllib.request.urlopen(f"{GATEWAY}/metrics", timeout=10).read().decode()
    out: dict[str, float] = {}
    for m in re.finditer(r'^gateway_requests_total\{cache="([^"]+)",status="(\d+)"\} (\S+)$', txt, re.M):
        out[m.group(1)] = out.get(m.group(1), 0) + float(m.group(3))
    return out


def restart_gateway(cache: bool) -> None:
    env = {**os.environ, "CACHE_ENABLED": "true" if cache else "false"}
    subprocess.run(
        ["docker", "compose", "up", "-d", "--force-recreate", "gateway"],
        check=True,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        ["docker", "compose", "exec", "-T", "redis", "redis-cli", "FLUSHALL"], check=True, capture_output=True
    )
    for _ in range(30):
        try:
            urllib.request.urlopen(f"{GATEWAY}/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("gateway did not come up")


def attack(targets: list[dict], rate: int, name: str) -> dict:
    vegeta = shutil.which("vegeta") or str(Path.home() / "go/bin/vegeta")
    tfile = SCRATCH / "targets.jsonl"
    tfile.write_text("\n".join(json.dumps(t) for t in targets))
    duration = f"{len(targets) / rate:.0f}s"
    res = SCRATCH / f"{name}.bin"
    with res.open("wb") as f:
        subprocess.run(
            [
                vegeta,
                "attack",
                "-format=json",
                f"-targets={tfile}",
                f"-rate={rate}/s",
                f"-duration={duration}",
                "-timeout=120s",
                "-max-workers=64",
            ],
            check=True,
            stdout=f,
        )
    rep = json.loads(
        subprocess.run(
            [vegeta, "report", "-type=json", str(res)], check=True, capture_output=True, text=True
        ).stdout
    )
    lat = rep["latencies"]
    return {
        "requests": rep["requests"],
        "rate_per_s": rate,
        "success_ratio": rep["success"],
        "latency_ms": {k: round(lat[k] / 1e6, 1) for k in ("50th", "90th", "95th", "99th", "max", "mean")},
        "status_codes": rep["status_codes"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--rate", type=int, default=2)  # below model capacity (warm miss ~0.7-1.2 s)
    args = ap.parse_args()
    SCRATCH.mkdir(parents=True, exist_ok=True)
    targets = build_targets(args.n, args.repeat)
    out = {
        "distinct_questions": args.n,
        "repeat": args.repeat,
        "requests_per_run": len(targets),
        "max_possible_hit_rate": round(1 - args.n / len(targets), 3),
        "runs": {},
    }
    for name, cache in (("no_cache", False), ("with_cache", True)):
        restart_gateway(cache)
        before = metrics()
        r = attack(targets, args.rate, name)
        after = metrics()
        delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
        hits = delta.get("hit_exact", 0) + delta.get("hit_normalized", 0)
        total = sum(delta.values())
        r["cache_counts"] = {k: int(v) for k, v in delta.items() if v}
        r["hit_rate"] = round(hits / total, 3) if total else 0.0
        out["runs"][name] = r
        print(name, json.dumps(r))
    Path("results/loadtest.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
