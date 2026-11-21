"""Evaluate a model (base zero-shot / base few-shot / QLoRA adapter) by execution accuracy.

Usage:
  uv run python -m sqlforge.eval.run --tag base_zeroshot --mode zeroshot
  uv run python -m sqlforge.eval.run --tag base_fewshot  --mode fewshot
  uv run python -m sqlforge.eval.run --tag qlora --adapter runs/qlora/adapter
Writes results/eval_<tag>_<split>.json and data/eval/<tag>_<split>.parquet (predictions).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import polars as pl
import torch

from sqlforge.eval.score import Scorer, extract_sql, summarize
from sqlforge.train.qlora import DEFAULT_BASE_MODEL, SYSTEM_PROMPT, load_base

DATA_DIR = Path("data/dataset")
PRED_DIR = Path("data/eval")
N_SHOTS = 3


def user_msg(ddl: str, question: str) -> str:
    return f"{ddl}\n\nQuestion: {question}"


def fewshot_examples(n: int = N_SHOTS) -> list[dict]:
    """Fixed exemplars from train (original names, short DDL, distinct kinds, both languages)."""
    df = pl.read_parquet(DATA_DIR / "train.parquet").filter(~pl.col("renamed"))
    df = df.with_columns(pl.col("ddl").str.len_chars().alias("_len")).filter(pl.col("_len") < 450)
    picks = []
    for kind, lang in [("group_agg", "en"), ("count_where", "de"), ("topk", "en")][:n]:
        cand = df.filter((pl.col("kind") == kind) & (pl.col("lang") == lang) & (pl.col("num_joins") <= 1))
        picks.append(cand.sort("id").row(0, named=True))
    return picks


def build_messages(row: dict, shots: list[dict]) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for s in shots:
        msgs.append({"role": "user", "content": user_msg(s["ddl"], s["question"])})
        msgs.append({"role": "assistant", "content": s["sql"]})
    msgs.append({"role": "user", "content": user_msg(row["ddl"], row["question"])})
    return msgs


@torch.inference_mode()
def generate(model, tok, prompts: list[str], batch_size: int, max_new_tokens: int = 256) -> list[str]:
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # less padding
    out: list[str] = [""] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b : b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id
        )
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
        for i, t in zip(idx, texts, strict=True):
            out[i] = t
    return out


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--mode", choices=["zeroshot", "fewshot"], default="zeroshot")
    ap.add_argument("--model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--splits", nargs="*", default=["test_in", "test_ood"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--source",
        choices=["final", "template"],
        default="final",
        help="template = use un-paraphrased template questions (ablation / smoke test)",
    )
    args = ap.parse_args()

    torch.manual_seed(0)
    model, tok = load_base(args.model)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    model.config.use_cache = True
    shots = fewshot_examples() if args.mode == "fewshot" else []
    scorer = Scorer()
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        if args.source == "template":
            df = pl.read_parquet(DATA_DIR / f"{split}_template.parquet").with_columns(
                pl.col("question_template").alias("question"), pl.lit("template").alias("question_source")
            )
        else:
            df = pl.read_parquet(DATA_DIR / f"{split}.parquet")
        if args.limit:
            df = df.head(args.limit)
        rows = df.to_dicts()
        prompts = [
            tok.apply_chat_template(build_messages(r, shots), tokenize=False, add_generation_prompt=True)
            for r in rows
        ]
        t0 = time.time()
        outputs = generate(model, tok, prompts, args.batch_size)
        gen_s = time.time() - t0
        for r, o in zip(rows, outputs, strict=True):
            r["raw_output"] = o
            r["pred_sql"] = extract_sql(o)
            r["outcome"] = scorer.score(r["pred_sql"], r["sql"], r["variant"], r["ordered"])
        summary = summarize(rows)
        summary.update(
            {
                "tag": args.tag,
                "split": split,
                "mode": args.mode,
                "model": args.model,
                "adapter": args.adapter,
                "n_shots": len(shots),
                "decoding": "greedy, max_new_tokens=256",
                "generation_seconds": round(gen_s, 1),
                "git_sha": git_sha(),
                "limit": args.limit,
                "question_source_mode": args.source,
            }
        )
        suffix = (f"_limit{args.limit}" if args.limit else "") + (
            "_tmpl" if args.source == "template" else ""
        )
        out_json = Path(f"results/eval_{args.tag}_{split}{suffix}.json")
        out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        pl.DataFrame(rows).select(
            "id", "split", "db", "kind", "lang", "question", "sql", "pred_sql", "raw_output", "outcome"
        ).write_parquet(PRED_DIR / f"{args.tag}_{split}{suffix}.parquet")
        print(
            f"{args.tag} {split}: EX={summary['execution_accuracy']:.3f} "
            f"valid={summary['valid_sql_rate']:.3f} "
            f"({gen_s:.0f}s) {summary['error_taxonomy']}"
        )
        _log_mlflow(summary, args)


def _log_mlflow(summary: dict, args) -> None:
    try:
        import mlflow

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        mlflow.set_experiment("sqlforge-eval")
        with mlflow.start_run(run_name=f"{summary['tag']}_{summary['split']}"):
            mlflow.log_params(
                {
                    "tag": args.tag,
                    "mode": args.mode,
                    "model": args.model,
                    "adapter": args.adapter,
                    "split": summary["split"],
                    "limit": args.limit,
                    "git_sha": summary["git_sha"],
                }
            )
            mlflow.log_metrics(
                {k: summary[k] for k in ("execution_accuracy", "valid_sql_rate", "select_only_rate")}
            )
    except Exception as e:  # noqa: BLE001  (tracking must never break an eval)
        print(f"mlflow logging skipped: {e}")


if __name__ == "__main__":
    main()
