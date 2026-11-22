"""Merge the QLoRA adapter into the bf16 base, quantize to GGUF Q4_K_M via Ollama, export for serving.

Steps: base (bf16, CPU) + adapter -> merge_and_unload -> runs/merged (safetensors)
       -> llama.cpp convert_hf_to_gguf.py (pinned tag, lazy tensors, ~4 GB RAM) -> F16 GGUF
       -> `ollama create <name> -q q4_K_M` from the F16 GGUF (host Ollama, ~18 GB RAM peak)
       -> copy the quantized blob + Modelfile to runs/serve/
Ollama's own safetensors converter is not used: on a 7B bf16 model it peaked at ~30 GB RSS and was OOM-killed.
The compose `ollama` service registers runs/serve/Modelfile as model "sqlforge" on start.

Usage: uv run python -m sqlforge.serve.export --adapter runs/qlora/adapter [--name sqlforge]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from sqlforge.train.qlora import DEFAULT_BASE_MODEL

# Qwen2.5 ChatML chat template (Ollama Go-template syntax). Ollama does not derive it from safetensors.
CHATML = """{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 -}}
<|im_start|>{{ .Role }}
{{ .Content }}<|im_end|>
{{ if and (ne .Role "assistant") $last }}<|im_start|>assistant
{{ end }}
{{- end }}"""

MODELFILE_TAIL = f'''TEMPLATE """{CHATML}"""
PARAMETER temperature 0
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER num_ctx 4096
'''


def merge(adapter: Path, out: Path, base: str) -> None:
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": "cpu"})
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(out)
    patch_legacy_config(out / "config.json")


def patch_legacy_config(path: Path) -> None:
    """transformers 5 writes `rope_parameters.rope_theta`; GGUF converters (Ollama) read top-level
    `rope_theta` and otherwise get freq_base=0 -> the runner aborts. Write the legacy keys too."""
    cfg = json.loads(path.read_text())
    rp = cfg.get("rope_parameters") or {}
    if "rope_theta" not in cfg and "rope_theta" in rp:
        cfg["rope_theta"] = rp["rope_theta"]
    if rp.get("rope_type", "default") == "default":
        cfg.setdefault("rope_scaling", None)
    cfg.setdefault("torch_dtype", cfg.get("dtype", "bfloat16"))
    path.write_text(json.dumps(cfg, indent=2))


LLAMA_CPP_TAG = "v1.0.0"  # b1b770d90d94cad9e9ea5b993bdbc383a522fb28
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", Path.home() / ".cache/sqlforge/llama.cpp"))


def convert_f16(merged: Path) -> Path:
    """safetensors -> F16 GGUF with llama.cpp's converter (needs only its gguf-py, no build)."""
    if not LLAMA_CPP_DIR.exists():
        LLAMA_CPP_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(
            [
                "git",
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                LLAMA_CPP_TAG,
                "https://github.com/ggml-org/llama.cpp",
                str(LLAMA_CPP_DIR),
            ]
        )
    out = merged / "model-f16.gguf"
    env = {**os.environ, "PYTHONPATH": str(LLAMA_CPP_DIR / "gguf-py")}
    run(
        [
            sys.executable,
            str(LLAMA_CPP_DIR / "convert_hf_to_gguf.py"),
            str(merged),
            "--outtype",
            "f16",
            "--outfile",
            str(out),
        ],
        env=env,
    )
    return out


def run(cmd: list[str], **kw) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="runs/qlora/adapter")
    ap.add_argument("--base", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--name", default="sqlforge")
    ap.add_argument("--merged", default="runs/merged")
    ap.add_argument("--serve-dir", default="runs/serve")
    ap.add_argument("--skip-merge", action="store_true", help="reuse an existing merged directory")
    args = ap.parse_args()
    merged, serve = Path(args.merged).resolve(), Path(args.serve_dir)

    if not args.skip_merge:
        merge(Path(args.adapter), merged, args.base)
    patch_legacy_config(merged / "config.json")
    f16 = convert_f16(merged)
    (merged / "Modelfile").write_text(f"FROM {f16}\n{MODELFILE_TAIL}")
    run(["ollama", "create", args.name, "-q", "q4_K_M", "-f", str(merged / "Modelfile")])

    modelfile = run(["ollama", "show", "--modelfile", args.name])
    blob = Path(re.search(r"^FROM (\S+)", modelfile, re.M).group(1))
    serve.mkdir(parents=True, exist_ok=True)
    gguf = serve / f"{args.name}-q4_k_m.gguf"
    try:
        shutil.copyfile(blob, gguf)
    except PermissionError:  # systemd Ollama stores blobs as user `ollama`
        run(["sudo", "-n", "cp", str(blob), str(gguf)])
    (serve / "Modelfile").write_text(f"FROM ./{gguf.name}\n{MODELFILE_TAIL}")
    sha = hashlib.sha256(gguf.read_bytes()).hexdigest()
    meta = {
        "adapter": args.adapter,
        "base": args.base,
        "gguf": str(gguf),
        "quant": "Q4_K_M",
        "gguf_bytes": gguf.stat().st_size,
        "gguf_sha256": sha,
    }
    (serve / "export.json").write_text(json.dumps(meta, indent=2))
    Path("results/export_gguf.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
