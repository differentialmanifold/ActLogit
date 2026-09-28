from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from actlogit.schema import StrictModel


class ModelConfig(StrictModel):
    name_or_path: str
    backend: Literal["transformers", "mlx"] = "transformers"
    adapter_path: str | None = None
    revision: str | None = None
    device: str = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    trust_remote_code: bool = False
    max_prompt_tokens: Annotated[int | None, Field(gt=0)] = None


class PromptConfig(StrictModel):
    # None discovers tokenizer-compatible labels; explicit labels must each be one token.
    labels: list[str] | None = None

    def signature(self) -> dict:
        # Record the fixed thinking mode alongside the configurable labels.
        return {**self.model_dump(), "enable_thinking": False}


def prompt_token_limit(configured: int | None, metadata: dict, tokenizer) -> int:
    """Use the smallest explicit or advertised context limit, ignoring HF sentinels."""
    limits = [configured] if configured is not None else []
    text = metadata.get("text_config") or {}
    for source in (metadata, text):
        for key in ("max_position_embeddings", "n_positions", "max_seq_len"):
            value = source.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and 0 < value < 10**9:
                limits.append(value)
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, int) and 0 < value < 10**9:
        limits.append(value)
    if not limits:
        raise ValueError("model context limit is unknown; set model.max_prompt_tokens explicitly")
    return min(limits)


class LoRAConfig(StrictModel):
    rank: Annotated[int, Field(gt=0)] = 64
    alpha: Annotated[int, Field(gt=0)] = 128
    dropout: Annotated[float, Field(ge=0, lt=1)] = 0.0
    target_modules: str | list[str] = "all-linear"
    num_layers: Annotated[int | None, Field(gt=0)] = None


class TrainingConfig(StrictModel):
    data: str
    output_dir: str
    eval_data: str | None = None
    epochs: Annotated[int, Field(gt=0)] = 3
    batch_size: Annotated[int, Field(gt=0)] = 4
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    learning_rate: Annotated[float, Field(gt=0)] = 5e-4
    weight_decay: Annotated[float, Field(ge=0)] = 0.0
    max_grad_norm: Annotated[float, Field(gt=0)] = 1.0
    max_steps: Annotated[int | None, Field(gt=0)] = None
    gradient_checkpointing: bool = False
    seed: int = 42


class Config(StrictModel):
    model: ModelConfig
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingConfig | None = None


def load_config(path: str | Path) -> Config:
    with Path(path).open("rb") as handle:
        return Config.model_validate(tomllib.load(handle))
