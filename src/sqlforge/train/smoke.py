"""Day-1 GPU smoke test: 50-step QLoRA on a tiny synthetic dataset, logs peak VRAM to results/smoke.json.

Usage: uv run python -m sqlforge.train.smoke [--model ID] [--steps 50] [--seq-len 1024]
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from sqlforge.train.qlora import DEFAULT_BASE_MODEL, load_base, lora_config, make_example

DDL = """CREATE TABLE nation (n_nationkey INTEGER, n_name VARCHAR, n_regionkey INTEGER);
CREATE TABLE customer (c_custkey INTEGER, c_name VARCHAR, c_nationkey INTEGER, c_acctbal DECIMAL(15,2));"""

PAIRS = [
    ("How many customers are there?", "SELECT COUNT(*) FROM customer;"),
    ("Wie viele Kunden gibt es?", "SELECT COUNT(*) FROM customer;"),
    ("List all nation names.", "SELECT n_name FROM nation;"),
    ("Zeige alle Ländernamen.", "SELECT n_name FROM nation;"),
    (
        "Average account balance per nation?",
        "SELECT n.n_name, AVG(c.c_acctbal) FROM customer c JOIN nation n "
        "ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name;",
    ),
    (
        "Durchschnittlicher Kontostand pro Land?",
        "SELECT n.n_name, AVG(c.c_acctbal) FROM customer c JOIN nation n "
        "ON c.c_nationkey = n.n_nationkey GROUP BY n.n_name;",
    ),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--pad-to-max", action="store_true", help="pad every sample to seq-len (worst-case VRAM)")
    ap.add_argument("--out", default="results/smoke.json")
    args = ap.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()

    # Long filler in the DDL so --pad-to-max actually measures ~seq-len tokens per sample.
    filler = (
        ""
        if not args.pad_to_max
        else "\n".join(
            f"CREATE TABLE t{i} (id INTEGER, val_{i} DOUBLE, note_{i} VARCHAR);" for i in range(40)
        )
    )
    rows = [make_example(DDL + "\n" + filler, q, s) for q, s in PAIRS * 40]
    ds = Dataset.from_list(rows)

    t0 = time.time()
    model, tok = load_base(args.model)
    load_s = time.time() - t0

    cfg = SFTConfig(
        output_dir="/tmp/sqlforge-smoke",
        max_steps=args.steps,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # transformers 5: float = ratio
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        max_length=args.seq_len,
        completion_only_loss=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        seed=0,
    )
    trainer = SFTTrainer(
        model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora_config()
    )
    seq_lens = [len(x) for x in trainer.train_dataset["input_ids"]]
    t1 = time.time()
    out = trainer.train()
    train_s = time.time() - t1

    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    res = {
        "model": args.model,
        "steps": args.steps,
        "max_length": args.seq_len,
        "effective_batch": 16,
        "sample_tokens_max": max(seq_lens),
        "sample_tokens_mean": round(sum(seq_lens) / len(seq_lens), 1),
        "peak_vram_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_vram_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "load_seconds": round(load_s, 1),
        "train_seconds": round(train_s, 1),
        "seconds_per_step": round(train_s / args.steps, 2),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "train_loss": out.training_loss,
        "gpu": torch.cuda.get_device_name(0),
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            **{p: version(p) for p in ("transformers", "peft", "trl", "bitsandbytes", "accelerate")},
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
