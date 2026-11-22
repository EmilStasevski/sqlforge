.PHONY: gguf artifacts eval-qlora env-check data data-push data-schemas smoke train eval eval-baseline results-md serve up down test lint bench loadtest

PY := uv run python

env-check:
	$(PY) scripts/env_check.py --train

data-schemas:
	$(PY) -m sqlforge.schemas.build all

# One command: schemas -> verified dataset -> paraphrase (DVC skips unchanged stages). Needs Ollama for paraphrase.
data:
	uv run dvc repro

data-push:
	docker compose up -d minio
	uv run dvc push

smoke:
	$(PY) -m sqlforge.train.smoke --out results/smoke_7b_short.json
	$(PY) -m sqlforge.train.smoke --pad-to-max --out results/smoke_7b_seq1024.json

train:
	$(PY) -m sqlforge.train.run --epochs 1

gguf:
	$(PY) -m sqlforge.serve.export --adapter runs/qlora/adapter

artifacts:
	docker compose up -d minio
	$(PY) scripts/push_artifacts.py

eval-baseline:
	$(PY) -m sqlforge.eval.run --tag base_zeroshot --mode zeroshot
	$(PY) -m sqlforge.eval.run --tag base_fewshot --mode fewshot
	$(PY) scripts/make_results_md.py

eval-qlora:
	$(PY) -m sqlforge.eval.run --tag qlora --adapter runs/qlora/adapter
	$(PY) scripts/make_results_md.py

eval: eval-baseline eval-qlora

results-md:
	$(PY) scripts/make_results_md.py

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

bench:
	$(PY) scripts/bench_sqlnorm.py

serve up:
	docker compose up -d --build

down:
	docker compose down

loadtest:
	$(PY) scripts/loadtest.py
