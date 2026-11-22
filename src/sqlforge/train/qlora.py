"""QLoRA building blocks shared by the smoke test and the full training run."""

from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from sqlforge.prompt import SYSTEM_PROMPT  # noqa: F401  (re-exported)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def bnb_config() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05) -> LoraConfig:
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
    )


def load_base(model_id: str = DEFAULT_BASE_MODEL):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config(),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model.config.use_cache = False
    return model, tok


def make_example(ddl: str, question: str, sql: str | None = None) -> dict:
    """Prompt/completion conversational record (TRL computes loss on completion only)."""
    rec = {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{ddl}\n\nQuestion: {question}"},
        ]
    }
    if sql is not None:
        rec["completion"] = [{"role": "assistant", "content": sql}]
    return rec
