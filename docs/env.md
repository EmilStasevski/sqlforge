# Environment (verified 2026-11-21)

All versions below are what `uv.lock` resolved and what the smoke test actually ran with
(the `env` block in `results/smoke_*.json` is the source of truth).

## Hardware / OS
- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB (Blackwell, compute capability 12.0 / sm_120)
- Driver 580.178.04, Linux 6.17, 31 GB RAM

## GPU training stack (`uv sync --group train`)
| Package | Version | Note |
|---|---|---|
| Python | 3.12.14 | pinned `>=3.12,<3.13` for wheel availability |
| torch | 2.11.0+cu128 | from `https://download.pytorch.org/whl/cu128`; `torch.cuda.get_arch_list()` includes `sm_120` |
| transformers | 5.24.0 | v5 API: `dtype=` instead of `torch_dtype=`; `warmup_ratio` removed → `warmup_steps=0.03` (float = ratio) |
| peft | 0.22.0 | `target_modules="all-linear"` |
| trl | 1.18.2 | `SFTConfig(max_length=..., completion_only_loss=True)` with prompt/completion conversational records |
| bitsandbytes | 0.50.2 | NF4 + double quant and `paged_adamw_8bit` work on sm_120 out of the box |
| accelerate | 1.16.0 | |
| mlflow / wandb | 3.18.0 / 0.31.2 | compose MLflow server image pinned to the same version (`v3.18.0`) |

No workarounds were needed for Blackwell beyond using the cu128 wheel index. `attn_implementation="sdpa"`, no flash-attn.

## Data / tooling
- duckdb 2.0.1 (tpch extension via `INSTALL tpch`), polars 2.1.0, pyarrow 26.0.1, pandera 0.36.0, sqlglot 30.33.0
- Rust 1.99.1, sqlparser 0.64.0, pyo3 0.30.1 (abi3-py312), maturin (build backend, via uv path dependency)
- Go 1.27.4, Docker 29.9.3, Ollama 0.13.5, uv 0.12.21
- dbt-core 1.12.7 + dbt-duckdb 1.11.1, DVC 3.67.1 (S3 remote = MinIO), boto3
- Paraphrase model: Ollama `qwen3:8b` (`think: false`). `qwen3:4b-instruct-2507` was tried first: on the same 120
  val questions it returned the template unchanged 81 % of the time, so only 18 % of its paraphrases were accepted (vs 77 % for qwen3:8b).

## Base model
- `Qwen/Qwen2.5-Coder-7B-Instruct` (snapshot `c03e6d358207e414f1eca0bb1891e29f1db0e242`), 4-bit NF4.

## Smoke test (`make smoke`)
Qwen2.5-Coder-7B-Instruct, NF4 + double quant, LoRA r=16 all-linear, batch 2 x accum 8, grad checkpointing, 50 steps.
Values copied from `results/smoke_7b_*.json`.

| Run | tokens/sample (max / mean) | peak VRAM alloc / reserved (GiB) | s/step | loss first → last |
|---|---|---|---|---|
| short samples | 121 / 97.7 | 5.61 / 5.67 | 4.7 | 0.0855 → 3.7e-05 |
| ~1k-token samples (`--pad-to-max`) | 931 / 907.7 | 6.4 / 6.52 | 19.95 | 0.0964 → 3.5e-05 |

=> 7B at max_length 1024 fits with ~9 GB headroom; no need to fall back to a 3-4B model.

## Docker images (tags checked with `docker manifest inspect` on 2026-11-21)
- `minio/minio` and `quay.io/minio/minio` no longer resolve for any tag tried (MinIO stopped publishing community
  images). Using `bitnamilegacy/minio:2025.7.23`, a frozen build of the MinIO server, which is fine for an
  S3-API artifact store.

## Gotchas
- **transformers 5 + GGUF:** merged checkpoints store `rope_parameters.rope_theta`; Ollama's safetensors importer
  reads the legacy top-level `rope_theta`, gets freq_base=0 and the runner aborts (SIGABRT) on first request.
  `serve/export.py` writes the legacy keys. Ollama also does not take the chat template from safetensors, so the
  Modelfile sets the Qwen ChatML template and stop tokens explicitly.
- **Do not let Ollama convert safetensors.** `ollama create -q q4_K_M` from the merged bf16 safetensors (7B, 15 GB)
  peaked at 29.9 GB RSS in the Ollama 0.13.5 server and was killed by the kernel OOM killer on a 31 GB host (kernel
  log: `Killed process ... (ollama) anon-rss:29913788kB`). `serve/export.py` therefore converts with llama.cpp's
  `convert_hf_to_gguf.py` (tag v1.0.0, lazy tensors, 3.9 GB peak RSS, needs `sentencepiece`) to an F16 GGUF and
  lets Ollama only quantize GGUF -> Q4_K_M (17.8 GB peak RSS). Before that, an earlier run next to a loaded 8B
  model had also OOM-killed the host Ollama (paraphrase requests in flight fell back to templates and were
  retried; failures are never cached).
- `pgrep -f <pattern>` inside a shell whose command line contains the same pattern matches that shell; wait on the
  PID of the actual python process (`ps -eo pid,args`) instead.
- Docker bridge DNS does not resolve on this host (router at 192.168.0.1 ignores the bridge subnet), so image
  builds use `network: host` and the Grafana ClickHouse plugin (4.23.5) is baked into the image.
- kind 0.29: `kind load docker-image` fails on multi-arch images ("content digest ... not found"); the deploy script
  imports via `ctr images import` for the current platform. The kind network is dual-stack: pick the IPv4 gateway.
- transformers 5: `apply_chat_template(tokenize=True)` returns a dict, not a list of ids.
- DuckDB opens one thread pool per connection; 8 worker processes x many cached connections hit the OS thread
  limit ("can't start new thread"). Fix: `SET threads = 2` per connection and at most 8 cached variant connections.
  (Passing `config={"threads": 2}` to `connect` fails if the same file is already open with a different config.)
- dbt-duckdb: `GROUP BY ALL` mis-detects aggregates inside `CASE ... avg(...) ... sum(...)`; use explicit ordinals.
- DuckDB `executemany` is row-by-row: inserting 105k weather rows took minutes. Arrow bulk insert (`con.register`) takes <1 s.
- `uv run` syncs inexactly, so the train group stays installed; `uv sync` without `--group train` removes it.
