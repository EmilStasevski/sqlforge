"""`make env-check`: verify toolchains, Python stack, GPU and built DBs. Exit 1 on required failure."""

from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_TOOLS = ["uv", "cargo", "go", "docker"]
OPTIONAL_TOOLS = ["ollama", "maturin", "dvc"]
REQUIRED_PY = ["duckdb", "polars", "pyarrow", "pandera"]
TRAIN_PY = ["torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "mlflow", "wandb"]
DBS = ["tpch", "smard", "weather"]

ok = True


def report(name: str, good: bool, info: str = "", required: bool = True) -> None:
    global ok
    mark = "OK  " if good else ("FAIL" if required else "warn")
    print(f"[{mark}] {name:<14} {info}")
    if required and not good:
        ok = False


def tool_version(t: str) -> str:
    args = [t, "version"] if t == "go" else [t, "--version"]
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"error: {e}"


for t in REQUIRED_TOOLS + OPTIONAL_TOOLS:
    found = shutil.which(t) is not None
    report(t, found, tool_version(t) if found else "missing", required=t in REQUIRED_TOOLS)

for m in REQUIRED_PY:
    try:
        report(m, True, importlib.import_module(m).__version__)
    except ImportError as e:
        report(m, False, str(e))

train = "--train" in sys.argv
for m in TRAIN_PY:
    try:
        report(m, True, importlib.import_module(m).__version__, required=train)
    except ImportError:
        report(m, False, "not installed (uv sync --group train)", required=train)

if train:
    import torch

    cuda = torch.cuda.is_available()
    info = (
        f"{torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
        if cuda
        else "no CUDA"
    )
    report("cuda", cuda, info)

for db in DBS:
    p = Path(f"data/db/{db}.duckdb")
    report(f"db:{db}", p.exists(), str(p) if p.exists() else "missing (make data-schemas)")

sys.exit(0 if ok else 1)
