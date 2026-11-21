# sqlforge — German/English text-to-SQL, QLoRA-tuned on execution-verified synthetic data

## Why this project exists
Emil Stasevski (M.Sc. Data Science, TU Dortmund) is applying for Werkstudent roles in Data Science / ML / Data Engineering in NRW and nearby.
His CV already shows: Python, Rust (PyO3, axum), Go, SQL, R, Polars, pandas, PySpark, PostgreSQL, ClickHouse, Parquet, PyTorch,
scikit-learn, GBMs, HF Transformers, ONNX, BayesFlow, Ollama/RAG, Docker, GitHub Actions, MLflow, DVC, Prometheus, Linux.

This one-week project closes the remaining gaps that job ads ask for, so the CV reads as "universal DS/ML/DE":
**LLM fine-tuning (QLoRA/PEFT/TRL/bitsandbytes), Redis, Kafka, FastAPI, DuckDB, dbt, Pandera, Grafana, MinIO (S3 API), W&B,
and (stretch) Airflow, Kubernetes, pgvector, Streamlit.** Azure is explicitly OUT of scope.
Do not duplicate what existing projects already cover (anomaly-inspect, Ruhrwatt, SBI proteins). Reuse, don't redo.

## Non-negotiable rules
1. **No manual data checking.** Every training label must be verified automatically: SQL is executed on DuckDB and kept only if it
   runs, returns a sane result, and passes the guards below. Natural-language paraphrases are kept only if they pass the automatic
   literal/term-preservation check. Never ask the user to eyeball samples.
2. **Never fabricate numbers.** Every metric in README/CV text must come from a file in `results/*.json` produced by code.
   Unknown values stay as `TODO(measure)`.
3. **Everything reproducible:** fixed seeds, pinned versions (`uv.lock`), `make` targets, DVC for data, MLflow for runs.
4. **Everything tested:** pytest for generator, verifier, Rust bindings, gateway. CI on GitHub Actions (ruff, pytest, cargo test, go test).
5. **Small, working increments.** Commit after each green step. Ask before any GPU job expected to exceed ~2 h.
6. Scope discipline: finish P0 → P1 → P2 in order. A shallow-but-real P1 beats a half-built P2. Only list on the CV what actually runs.

## Hardware / environment
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM (Blackwell, sm_120). Assume Linux or WSL2.
- Blackwell needs a recent PyTorch built for CUDA 12.8+ and a recent bitsandbytes. **Day 1 first task: smoke test** (50-step QLoRA on
  a tiny dataset, log peak VRAM). If something fails, find the fix, pin versions, and record them in `docs/env.md`. Do not guess versions from memory.
- Use `attn_implementation="sdpa"` (skip flash-attn). Gradient checkpointing on.
- Base model: a ~7-8B instruct/coder model in 4-bit NF4 (default: Qwen2.5-Coder-7B-Instruct; use the current equivalent if better supported).
  If 16 GB is too tight at seq len 1024, drop to a 3-4B model — the pipeline matters more than the size.

## Core idea (data is derived, not collected)
Schemas -> grammar-based SQL sampler -> execute on DuckDB (ground truth) -> NL question templates (DE + EN) -> local-LLM paraphrase
(Ollama) -> automatic filters -> dataset. Evaluation = **execution accuracy** on a held-out schema the model never saw in training.

### Schemas (all auto-obtainable)
| Role | Source |
|---|---|
| train | TPC-H via DuckDB (`INSTALL tpch; LOAD tpch; CALL dbgen(sf=0.1);`) — 8 tables, FKs, no download |
| train | SMARD electricity series — reuse Ruhrwatt Gold Parquet if present, else small download script; modelled with dbt (dbt-duckdb) |
| **held-out test** | Open-Meteo historical weather (free, no key) for ~6 NRW cities (Dortmund, Köln, Düsseldorf, Essen, Münster, Bonn) |

Schema augmentation (automatic): random table/column subsets, shuffled column order, alias/rename variants, so the model learns
to read the DDL in the prompt instead of memorising TPC-H.

### Prompt format
System: "Return a single DuckDB SELECT statement." + DDL of the (sub)schema + question (DE or EN) -> SQL only.

### Automatic verification pipeline (the "no manual check" guarantee)
A sample is kept only if ALL hold:
- Parses with the Rust `sqlnorm` crate; statement is a single SELECT (no DDL/DML, no multiple statements)
- Executes on read-only DuckDB within a timeout
- Result: 1 <= rows <= 10_000, not all-NULL, deterministic (run twice, same result; ORDER BY present when LIMIT is used)
- Canonical-SQL hash not seen before (dedup) and not in test split (leak check)
- Paraphrase keeps all literals (numbers, dates, strings) and glossary terms of the template question; otherwise fall back to the template question
- Pandera schema validates the final dataset table

Splits: train (TPC-H + SMARD), val (same schemas, disjoint hashes), test_in (same schemas), **test_ood (Open-Meteo)**.
Target size: ~10-15k train, 1k val, 1k test_in, 1k test_ood. Start with 2k for a dry run.

## Training (QLoRA)
TRL `SFTTrainer` + PEFT + bitsandbytes. Starting config (tune from there):
NF4 + double quant, bf16 compute, LoRA r=16 alpha=32 dropout=0.05 on all linear layers, lr 2e-4 cosine, warmup 3%,
max_seq_len 1024, per-device batch 2 x grad-accum 8, `paged_adamw_8bit`, loss on completion tokens only, 1-2 epochs.
Log to MLflow (primary) and Weights & Biases (offline-capable mirror). Save adapter + config + git SHA + data hash.

## Evaluation (all automatic, written to `results/*.json`)
- Execution accuracy (result-set equality; order-insensitive unless ORDER BY), valid-SQL rate, SELECT-only rate
- Per bucket: #joins, aggregation, group-by, nested subquery, language (DE/EN)
- Compare: base zero-shot vs base few-shot vs QLoRA, on test_in and test_ood
- Error taxonomy by script (syntax / wrong column / wrong aggregate / wrong result) — no manual labelling
- Rust `sqlnorm` benchmark vs `sqlglot` in Python (measured speedup only)
- Serving: p50/p95 latency and cache hit-rate with and without Redis (load test with `vegeta`)

## Architecture (docker compose)
```
client -> Go gateway (:8080) --Redis (exact + normalized-question cache, rate limit)
               |  \--Prometheus metrics (/metrics)
               |  \--Kafka producer: topic ask.events
               v
        FastAPI executor (:8000) -> llama.cpp server / Ollama (merged adapter, GGUF Q4_K_M)
               |-> Rust sqlnorm (PyO3): guard + normalize
               |-> DuckDB (read-only) executes SQL, returns rows
Kafka ask.events -> Python consumer -> ClickHouse -> Grafana dashboard (latency, hit-rate, error classes)
MinIO (S3 API): datasets (Parquet), adapters, eval reports      MLflow: runs/registry      Prometheus + Grafana
```
Serving note: merging LoRA -> GGUF -> llama.cpp/Ollama is the safe default on Blackwell; try vLLM only if time remains.

## Repo layout
```
sqlforge/
  CLAUDE.md  README.md  Makefile  pyproject.toml  uv.lock  dvc.yaml  docker-compose.yml
  docs/env.md  docs/results.md
  src/sqlforge/{schemas,sampler,nlgen,paraphrase,verify,dataset,train,eval,serve}/
  rust/sqlnorm/        # sqlparser-rs + PyO3 (maturin): normalize, is_safe_select, features, hash
  go/gateway/          # Redis, Prometheus, Kafka producer
  dbt/smard/           # dbt-duckdb models for SMARD marts
  dags/                # Airflow (P2)
  k8s/                 # kustomize manifests (P2)
  tests/  results/  .github/workflows/ci.yml
```
Makefile targets: `env-check data train eval serve up down test bench loadtest`.

## Priorities and 7-day plan (about 5-6 h/day; training runs overnight)
**P0 — must ship (Days 1-5):** QLoRA/PEFT/TRL/bitsandbytes, DuckDB, Rust `sqlnorm`, Pandera, MLflow + W&B, eval harness, uv, ruff, pytest, CI.
**P1 — Days 5-6:** Redis, Kafka, FastAPI, Go gateway, Grafana, MinIO, dbt, docker compose.
**P2 — only if time (Day 7):** Airflow DAG (`data -> train -> eval -> register`), kind + kustomize deploy, pgvector schema retrieval, Streamlit demo.

| Day | Goal | Done when |
|---|---|---|
| 1 | Scaffold repo, uv, CI skeleton, compose skeleton; GPU/QLoRA smoke test; load 3 schemas into DuckDB | smoke test logs VRAM; `make env-check` green; versions in docs/env.md |
| 2 | SQL sampler, DE/EN templates, Rust `sqlnorm` (parse/guard/normalize/hash), dbt SMARD marts, Pandera | sampler emits 2k valid queries; Rust tests + benchmark pass |
| 3 | Execution verifier, Ollama paraphrase + preservation filter, splits, DVC, push to MinIO; baseline eval (base model) | `data/` built from one command; baseline numbers in results/ |
| 4 | Full QLoRA run (start by midday, runs overnight); build FastAPI executor + Go gateway meanwhile | adapter + MLflow run + W&B run exist |
| 5 | Eval QLoRA vs baselines, error taxonomy; merge -> GGUF -> serve | results table generated by script |
| 6 | Redis cache, Kafka events -> ClickHouse -> Grafana; load test | dashboard shows live traffic; with/without-cache numbers saved |
| 7 | README (EN + short DE), architecture diagram, results table from JSON, P2 items if time, CV text | fresh clone + `make up` + `make eval` works |

## Definition of done
- [ ] One-command data build; labels verified by execution only
- [ ] QLoRA adapter beats base few-shot on test_ood execution accuracy (report honestly if not, with analysis)
- [ ] Rust speedup and Redis latency numbers measured by scripts
- [ ] `docker compose up` runs gateway, executor, model server, Redis, Kafka, ClickHouse, Grafana, MinIO, MLflow
- [ ] CI green; README has results tables generated from `results/*.json`
- [ ] CV bullets (below) filled with real numbers

## CV output (fill only from results/*.json)
Projekt (DE CV style): **sqlforge — Text-to-SQL (DE/EN) mit QLoRA** · PyTorch, PEFT/TRL, bitsandbytes, DuckDB, Rust, Go, Kafka, Redis
- Trainingsdaten vollständig automatisch erzeugt und per DuckDB-Ausführung verifiziert (N Beispiele, keine manuelle Prüfung); QLoRA auf einer 16-GB-GPU: Execution Accuracy X % statt Y % (Base, Few-Shot) auf einem unbekannten Schema.
- Rust-Modul (PyO3, sqlparser) zur SQL-Validierung/Deduplizierung: Z-mal schneller als sqlglot; Serving über Go-Gateway mit Redis-Cache (p95 A ms -> B ms), Kafka-Events -> ClickHouse -> Grafana.
Skills line to add after completion: LLM-Fine-Tuning (QLoRA, PEFT, TRL), Redis, Kafka, FastAPI, DuckDB, dbt, Grafana, MinIO/S3, Weights & Biases (+ Airflow, Kubernetes only if built).

## First actions for Claude Code
1. Read this file. Create the repo layout and `docs/env.md`. 
2. Run the Day-1 smoke test and report VRAM + working versions.
3. Create the task list for Day 1 and proceed; update `docs/results.md` only from real result files.
