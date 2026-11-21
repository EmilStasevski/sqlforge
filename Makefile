.PHONY: env-check data data-schemas smoke train eval serve up down test lint bench loadtest

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
	@echo "TODO(day4)"; exit 1

eval:
	@echo "TODO(day3/5)"; exit 1

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

test:
	uv run pytest -q

bench:
	$(PY) scripts/bench_sqlnorm.py

serve up:
	docker compose up -d

down:
	docker compose down

loadtest:
	@echo "TODO(day6): vegeta"; exit 1
