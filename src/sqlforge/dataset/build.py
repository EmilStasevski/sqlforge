"""Build the verified text-to-SQL dataset: sample -> verify -> template question -> dedup -> splits.

  test_ood  weather (held-out schema), full schema, original names
  test_in   tpch + smard, full schema, original names
  val       tpch + smard, augmented schemas (like train)
  train     tpch + smard, augmented schemas (25% full schema)

Dedup and leak check use the canonical hash (Rust sqlnorm) of the SQL on the original schema; splits are
generated test-first so every later split excludes all earlier hashes.

Usage: uv run python -m sqlforge.dataset.build [--scale 1.0] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl
import sqlnorm

from sqlforge.dataset.schema import TemplateSchema, assert_no_leak
from sqlforge.nlgen.render import render_question
from sqlforge.sampler.generate import Sampler
from sqlforge.sampler.spec import render_sql
from sqlforge.schemas.card import load_card
from sqlforge.schemas.variant import full_variant, random_variant
from sqlforge.verify.execute import Verifier

OUT_DIR = Path("data/dataset")
SPLITS: list[tuple[str, dict[str, float], int, bool]] = [
    # (name, db weights, target size at scale 1, augment schema)
    ("test_ood", {"weather": 1.0}, 1000, False),
    ("test_in", {"tpch": 0.6, "smard": 0.4}, 1000, False),
    ("val", {"tpch": 0.6, "smard": 0.4}, 1000, True),
    ("train", {"tpch": 0.6, "smard": 0.4}, 12000, True),
]
P_FULL_SCHEMA = 0.25
P_GERMAN = 0.5
CHUNK = 250

_verifiers: dict[str, Verifier] = {}


def _make(split: str, db: str, seed: int, augment: bool) -> dict | None:
    rng = random.Random(f"{split}-{db}-{seed}")
    card = load_card(db)
    spec = Sampler(card, rng).sample()
    if spec is None:
        return {"_reject": "none"}
    if augment and rng.random() >= P_FULL_SCHEMA:
        variant = random_variant(card, spec.columns(), rng, spec.tables())
    else:
        variant = full_variant(card)
    ver = _verifiers.setdefault(db, Verifier(card))
    verdict = ver.check(spec, variant)
    if not verdict.ok:
        return {"_reject": verdict.reason}
    lang = "de" if rng.random() < P_GERMAN else "en"
    q = render_question(spec, card, lang, rng)
    sql_orig = render_sql(spec, full_variant(card))
    sql = render_sql(spec, variant)
    feats = sqlnorm.features(sql_orig)
    return {
        "split": split,
        "db": db,
        "seed": seed,
        "kind": spec.kind,
        "lang": lang,
        "question_template": q.question,
        "literals": json.dumps(q.literals, ensure_ascii=False),
        "terms": json.dumps(q.terms, ensure_ascii=False),
        "sql": sql,
        "sql_orig": sql_orig,
        "sql_hash": sqlnorm.canonical_hash(sql_orig),
        "ordered": spec.order_by is not None,
        "n_rows": verdict.n_rows,
        "renamed": not variant.is_identity,
        "variant": variant.to_json(),
        "ddl": variant.ddl(),
        **{k: int(v) if isinstance(v, bool) else v for k, v in feats.items()},
    }


def _chunk(args: tuple[str, str, int, int, bool]) -> list[dict]:
    split, db, start, n, augment = args
    return [r for s in range(start, start + n) if (r := _make(split, db, s, augment)) is not None]


def build(scale: float, workers: int) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    frames: dict[str, pl.DataFrame] = {}
    stats: dict = {"scale": scale, "splits": {}}
    with ProcessPoolExecutor(workers) as pool:
        for split, weights, size, augment in SPLITS:
            t0 = time.time()
            target = max(1, round(size * scale))
            rows: list[dict] = []
            rejects: Counter = Counter()
            dup = 0
            for db, w in weights.items():
                want = round(target * w)
                got: list[dict] = []
                start = 0
                while len(got) < want:
                    jobs = [(split, db, start + i * CHUNK, CHUNK, augment) for i in range(workers)]
                    start += workers * CHUNK
                    for res in pool.map(_chunk, jobs):  # ordered by seed -> deterministic
                        for r in res:
                            if "_reject" in r:
                                rejects[r["_reject"]] += 1
                            elif r["sql_hash"] in seen:
                                dup += 1
                            elif len(got) < want:
                                seen.add(r["sql_hash"])
                                got.append(r)
                    if start > 200 * want + 10_000:
                        raise RuntimeError(f"{split}/{db}: cannot reach {want} unique samples")
                rows += got
            df = pl.DataFrame(rows).with_row_index("idx")
            df = df.with_columns(pl.format("{}-{}", pl.lit(split), pl.col("idx")).alias("id")).drop("idx")
            TemplateSchema.validate(df)
            frames[split] = df
            df.write_parquet(OUT_DIR / f"{split}_template.parquet")
            stats["splits"][split] = {
                "n": df.height,
                "by_db": dict(Counter(df["db"])),
                "by_lang": dict(Counter(df["lang"])),
                "by_kind": dict(sorted(Counter(df["kind"]).items())),
                "renamed_share": round(df["renamed"].mean(), 3),
                "rejected": dict(rejects),
                "duplicates_dropped": dup,
                "seconds": round(time.time() - t0, 1),
            }
            print(split, json.dumps(stats["splits"][split]))
    assert_no_leak(frames)
    stats["leak_check"] = "passed"
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    stats = build(args.scale, args.workers)
    Path("results").mkdir(exist_ok=True)
    Path("results/dataset_build.json").write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
