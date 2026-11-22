"""Full QLoRA SFT run (TRL SFTTrainer + PEFT + bitsandbytes NF4).

Logs to MLflow (primary, sqlite store or $MLFLOW_TRACKING_URI) and Weights & Biases (offline mirror unless
WANDB_MODE is set). Saves adapter + run_meta.json (config, git SHA, data hashes) and results/train_qlora.json.

Usage: uv run python -m sqlforge.train.run [--epochs 1] [--out runs/qlora] [--max-steps N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import polars as pl
import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from sqlforge.train.qlora import DEFAULT_BASE_MODEL, load_base, lora_config, make_example

DATA = Path("data/dataset")
SEED = 0


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def records(split: str, limit: int | None = None) -> list[dict]:
    df = pl.read_parquet(DATA / f"{split}.parquet")
    if limit:
        df = df.sample(n=min(limit, df.height), seed=SEED)
    return [make_example(d, q, s) for d, q, s in df.select("ddl", "question", "sql").iter_rows()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", default="runs/qlora")
    ap.add_argument("--eval-samples", type=int, default=300)
    args = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "sqlforge-train")
    os.environ.setdefault("WANDB_MODE", "offline")
    os.environ.setdefault("WANDB_PROJECT", "sqlforge")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {
        "base_model": args.model,
        "git_sha": git_sha,
        "data_sha256": {s: sha256(DATA / f"{s}.parquet") for s in ("train", "val")},
        "seed": SEED,
    }

    train_ds = Dataset.from_list(records("train")).shuffle(seed=SEED)
    eval_ds = Dataset.from_list(records("val", args.eval_samples))
    model, tok = load_base(args.model)
    torch.cuda.reset_peak_memory_stats()

    cfg = SFTConfig(
        output_dir=str(out / "checkpoints"),
        run_name=out.name,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=8,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # transformers 5: float = ratio
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=1024,
        completion_only_loss=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=150,
        save_strategy="steps",
        save_steps=150,
        save_total_limit=2,
        report_to=["mlflow", "wandb"],
        seed=SEED,
    )
    meta["config"] = {k: v for k, v in cfg.to_dict().items() if isinstance(v, (int, float, str, bool))}
    meta["lora"] = {"r": args.rank, "alpha": 2 * args.rank, "dropout": 0.05, "target_modules": "all-linear"}
    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tok,
        peft_config=lora_config(r=args.rank, alpha=2 * args.rank),
    )
    t0 = time.time()
    result = trainer.train()
    train_s = time.time() - t0
    final_eval = trainer.evaluate()

    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    hist = trainer.state.log_history
    res = {
        **meta,
        "steps": trainer.state.global_step,
        "epochs": round(trainer.state.epoch or 0, 3),
        "train_samples": len(train_ds),
        "train_loss": round(result.training_loss, 5),
        "eval_loss_final": round(final_eval["eval_loss"], 5),
        "eval_loss_curve": [(h["step"], round(h["eval_loss"], 5)) for h in hist if "eval_loss" in h],
        "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / max(1, trainer.state.global_step), 2),
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "adapter_dir": str(adapter_dir),
        "versions": {p: version(p) for p in ("torch", "transformers", "peft", "trl", "bitsandbytes")},
    }
    (adapter_dir / "run_meta.json").write_text(json.dumps(res, indent=2))
    Path("results/train_qlora.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
