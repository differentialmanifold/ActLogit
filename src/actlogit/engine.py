from __future__ import annotations

import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from actlogit.config import Config
from actlogit.prompt import TokenLabels, render_prompt
from actlogit.schema import DecisionRequest, DecisionResponse


@dataclass
class EncodedDecision:
    input_ids: list[int]
    candidate_ids: tuple[int, ...]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DecisionEngine:
    def __init__(self, model: Any, tokenizer: Any, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer needs a pad_token or eos_token")
            tokenizer.pad_token = tokenizer.eos_token
        self.codec = TokenLabels.build(tokenizer, config.prompt.labels)
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        parameters = inspect.signature(base.forward).parameters
        # Models supporting this avoid materializing vocab logits for the entire prompt.
        self._last_logit_arg = next(
            (name for name in ("logits_to_keep", "num_logits_to_keep") if name in parameters), None
        )
        self.model.eval()

    @classmethod
    def load(cls, config: Config, *, trainable_adapter: bool = False) -> DecisionEngine:
        if config.model.backend == "mlx":
            from actlogit.mlx_backend import MLXDecisionEngine

            return MLXDecisionEngine.load(config, trainable_adapter=trainable_adapter)
        from transformers import AutoModelForCausalLM, AutoTokenizer

        settings = config.model
        adapter = Path(settings.adapter_path) if settings.adapter_path else None
        manifest = None
        if adapter:
            manifest_path = adapter / "actlogit.json"
            if not manifest_path.is_file():
                raise ValueError(
                    f"adapter is missing its ActLogit prompt/token manifest: {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("format_version") != 1:
                raise ValueError("unsupported adapter manifest version")
            if (
                manifest["base_model"] != settings.name_or_path
                or manifest["base_revision"] != settings.revision
            ):
                raise ValueError(
                    "adapter requires the same base model and revision used for training"
                )
            if manifest["prompt"] != config.prompt.model_dump():
                raise ValueError(
                    "prompt configuration differs from the adapter training configuration"
                )
        device = resolve_device(settings.device)
        if settings.dtype == "auto":
            dtype = (
                torch.bfloat16
                if device.startswith("cuda") and torch.cuda.is_bf16_supported()
                else torch.float16
                if device.startswith("cuda")
                else torch.float32
            )
        else:
            dtype = getattr(torch, settings.dtype)
        common = {
            "local_files_only": settings.local_files_only,
            "trust_remote_code": settings.trust_remote_code,
            "revision": settings.revision,
        }
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter) if adapter else settings.name_or_path,
            **{**common, "revision": None if adapter else settings.revision},
        )
        model = AutoModelForCausalLM.from_pretrained(
            settings.name_or_path, torch_dtype=dtype, **common
        )
        if getattr(model.config, "is_encoder_decoder", False):
            raise ValueError("the local backend requires a decoder-only causal language model")
        model.to(device)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable_adapter)
        engine = cls(model, tokenizer, config)
        if manifest:
            if (
                list(engine.codec.labels) != manifest["labels"]
                or list(engine.codec.token_ids) != manifest["token_ids"]
            ):
                raise ValueError("tokenizer label mapping differs from the trained adapter")
        return engine

    @property
    def device(self) -> torch.device:
        return self.model.get_input_embeddings().weight.device

    @property
    def model_id(self) -> str:
        return self.config.model.name_or_path

    def encode(self, request: DecisionRequest) -> EncodedDecision:
        prompt, candidates, add_special_tokens = render_prompt(
            request, self.tokenizer, self.codec, self.config.prompt
        )
        ids = self.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
        limit = self.config.model.max_prompt_tokens
        context_limit = getattr(self.model.config, "max_position_embeddings", None)
        if isinstance(context_limit, int) and context_limit > 0:
            limit = min(limit, context_limit)
        if not ids or len(ids) > limit:
            raise ValueError(
                f"decision prompt has {len(ids)} tokens; allowed 1..{limit}. "
                "Reduce the supplied state; prompts are never silently truncated."
            )
        return EncodedDecision(ids, candidates)

    def logits(self, encoded: list[EncodedDecision]) -> tuple[Tensor, Tensor]:
        if not encoded:
            raise ValueError("decision batch is empty")
        length = max(len(item.input_ids) for item in encoded)
        actions = max(len(item.candidate_ids) for item in encoded)
        input_ids = torch.full(
            (len(encoded), length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        candidate_ids = torch.zeros((len(encoded), actions), dtype=torch.long, device=self.device)
        candidate_mask = torch.zeros_like(candidate_ids, dtype=torch.bool)
        for index, item in enumerate(encoded):
            input_ids[index, -len(item.input_ids) :] = torch.tensor(
                item.input_ids, device=self.device
            )
            attention_mask[index, -len(item.input_ids) :] = 1
            candidate_ids[index, : len(item.candidate_ids)] = torch.tensor(
                item.candidate_ids, device=self.device
            )
            candidate_mask[index, : len(item.candidate_ids)] = True
        position_ids = attention_mask.cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        kwargs = {self._last_logit_arg: 1} if self._last_logit_arg else {}
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
            **kwargs,
        )
        # This is the next-token prediction at the prompt boundary; no generated text is needed.
        final_logits = outputs.logits[:, -1, :]
        return final_logits.gather(-1, candidate_ids), candidate_mask

    @torch.inference_mode()
    def predict_many(self, requests: list[DecisionRequest]) -> list[DecisionResponse]:
        self.model.eval()
        logits, mask = self.logits([self.encode(request) for request in requests])
        probabilities = logits.float().masked_fill(~mask, -torch.inf).softmax(dim=-1)
        if not torch.isfinite(probabilities).all():
            raise ValueError("model produced nonfinite decision probabilities")
        responses = []
        for request, row in zip(requests, probabilities.cpu().tolist(), strict=True):
            count = len(request.choices)
            index = max(range(count), key=lambda i: row[i])
            responses.append(
                DecisionResponse(
                    choice=request.choices[index].id,
                    probabilities={choice.id: row[i] for i, choice in enumerate(request.choices)},
                    confidence=row[index],
                    token=self.codec.labels[index],
                    model=self.model_id,
                )
            )
        return responses

    def predict(self, request: DecisionRequest) -> DecisionResponse:
        return self.predict_many([request])[0]
