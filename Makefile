.PHONY: env-check data data-schemas smoke train eval serve up down test lint bench loadtest

PY := uv run python

env-check:
	$(PY) scripts/env_check.py --train

data-schemas:
	$(PY) -m sqlforge.schemas.build all

data: data-schemas
	@echo "TODO(day2-3): sampler -> verify -> nlgen -> paraphrase -> splits"

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
	@echo "TODO(day2): sqlnorm vs sqlglot"; exit 1

serve up:
	docker compose up -d

down:
	docker compose down

loadtest:
	@echo "TODO(day6): vegeta"; exit 1
