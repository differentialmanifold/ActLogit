"""Local MLX single-token decisions and weighted forward-KL LoRA training.

Each micro-example uses its true sequence length, including hybrid recurrent
models where padding without a matching recurrent mask would change the state.
Batch size controls the accumulation window, not padded model execution.
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from actlogit.config import Config, prompt_token_limit
from actlogit.data import load_records, write_json
from actlogit.prompt import TokenLabels, render_prompt
from actlogit.schema import DecisionRequest, DecisionResponse


@dataclass
class EncodedDecision:
    input_ids: list[int]
    candidate_ids: tuple[int, ...]


def candidate_logits(model, encoded):
    inputs = mx.array([encoded.input_ids])
    # Qwen3.5's text backbone exposes normalized hidden states. Apply the vocab
    # head only at the decision boundary, avoiding [prompt_length, vocab] logits.
    if getattr(model, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}:
        language = model.language_model
        hidden = language.model(inputs)[:, -1:, :]
        if language.args.tie_word_embeddings:
            logits = language.model.embed_tokens.as_linear(hidden)
        else:
            logits = language.lm_head(hidden)
    else:
        logits = model(inputs)
    return logits[0, -1, mx.array(encoded.candidate_ids)].astype(mx.float32)


def loss_terms(logits, target):
    logp = logits - mx.logsumexp(logits)
    logq = mx.log(mx.maximum(target, 1e-30))
    ce = -mx.sum(target * logp)
    entropy = -mx.sum(target * logq)
    return ce - entropy, ce, entropy


def install_lora(model, adapter_config):
    from mlx_lm.tuner.utils import linear_to_lora_layers

    count = adapter_config["num_layers"]
    if not 1 <= count <= len(model.layers):
        raise ValueError(f"lora.num_layers must be in 1..{len(model.layers)}")
    model.freeze()
    linear_to_lora_layers(model, count, adapter_config["lora_parameters"])
    if not tree_flatten(model.trainable_parameters()):
        raise ValueError("LoRA target_modules matched no layers")


class MLXDecisionEngine:
    def __init__(self, model, tokenizer, config, *, model_metadata=None):
        self.model, self.tokenizer, self.config = model, tokenizer, config
        self._model_metadata = model_metadata or {}
        self.codec = TokenLabels.build(tokenizer, config.prompt.labels)
        self.model.eval()

    @classmethod
    def load(cls, config: Config, *, trainable_adapter=False):
        from mlx_lm import load

        settings = config.model
        path = Path(settings.name_or_path).expanduser()
        # This backend is explicitly local: never download an accidental repo ID.
        if not path.is_dir():
            raise ValueError(f"MLX requires an existing local model directory: {path}")
        if settings.revision is not None or settings.device not in {"auto", "gpu"}:
            raise ValueError("MLX uses local weights on Apple GPU; omit revision/device")
        if settings.dtype != "auto":
            raise ValueError("MLX preserves the checkpoint dtype; use dtype=auto")
        manifest = None
        if settings.adapter_path:
            manifest = json.loads((Path(settings.adapter_path) / "actlogit.json").read_text())
            if manifest.get("format_version") != 1 or manifest.get("backend") != "mlx":
                raise ValueError("adapter is not an ActLogit MLX adapter")
            if (
                manifest["base_model"] != settings.name_or_path
                or manifest["base_revision"] != settings.revision
                or manifest["prompt"] != config.prompt.signature()
            ):
                raise ValueError("adapter base model/revision/prompt configuration mismatch")
        model, tokenizer = load(
            str(path),
            tokenizer_config={
                "local_files_only": True,
                "trust_remote_code": settings.trust_remote_code,
            },
        )
        engine = cls(
            model, tokenizer, config, model_metadata=json.loads((path / "config.json").read_text())
        )
        if manifest:
            if manifest["labels"] != list(engine.codec.labels) or manifest["token_ids"] != list(
                engine.codec.token_ids
            ):
                raise ValueError("adapter tokenizer label mapping mismatch")
            adapter_path = Path(settings.adapter_path)
            adapter_config = json.loads((adapter_path / "adapter_config.json").read_text())
            install_lora(model, adapter_config)
            weights = mx.load(str(adapter_path / "adapters.safetensors"))
            expected = dict(tree_flatten(model.trainable_parameters()))
            if set(weights) != set(expected) or any(
                weights[key].shape != value.shape for key, value in expected.items()
            ):
                raise ValueError("adapter weight keys/shapes do not match its configuration")
            model.load_weights(list(weights.items()), strict=False)
            mx.eval(model.parameters())
            model.eval()
        return engine

    @property
    def model_id(self):
        return self.config.model.name_or_path

    @property
    def max_prompt_tokens(self):
        return prompt_token_limit(
            self.config.model.max_prompt_tokens, self._model_metadata, self.tokenizer
        )

    def encode(self, request: DecisionRequest):
        prompt, candidates, special = render_prompt(request, self.tokenizer, self.codec)
        ids = self.tokenizer.encode(prompt, add_special_tokens=special)
        limit = self.max_prompt_tokens
        if not ids or len(ids) > limit:
            raise ValueError(f"decision prompt has {len(ids)} tokens; allowed 1..{limit}")
        return EncodedDecision(ids, candidates)

    def predict_many(self, requests):
        if not requests:
            raise ValueError("decision batch is empty")
        self.model.eval()
        responses = []
        for request in requests:
            values = mx.softmax(candidate_logits(self.model, self.encode(request))).tolist()
            if not all(math.isfinite(value) for value in values):
                raise ValueError("model produced nonfinite decision probabilities")
            index = max(range(len(values)), key=values.__getitem__)
            responses.append(
                DecisionResponse(
                    choice=request.choices[index].id,
                    probabilities={
                        choice.id: value
                        for choice, value in zip(request.choices, values, strict=True)
                    },
                    confidence=values[index],
                    token=self.codec.labels[index],
                    model=self.model_id,
                )
            )
        return responses

    def predict(self, request):
        return self.predict_many([request])[0]


def evaluate_records(engine, records, batch_size=4):
    if not records or batch_size < 1:
        raise ValueError("evaluation needs records and a positive batch size")
    was_training = engine.model.training
    engine.model.eval()
    totals = dict(forward_kl=0.0, cross_entropy=0.0, target_entropy=0.0, top1_agreement=0.0)
    try:
        for record in records:
            logits = candidate_logits(engine.model, engine.encode(record.decision))
            target = record.distribution()
            kl, ce, entropy = loss_terms(logits, mx.array(target))
            index = int(mx.argmax(logits).item())
            values = [
                kl.item(),
                ce.item(),
                entropy.item(),
                float(math.isclose(target[index], max(target), rel_tol=1e-5, abs_tol=1e-8)),
            ]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite evaluation metrics")
            for key, value in zip(totals, values, strict=True):
                totals[key] += value * record.weight
    finally:
        engine.model.train(was_training)
    denominator = math.fsum(record.weight for record in records)
    return {key: value / denominator for key, value in totals.items()}


def train(config: Config, *, log=True):
    settings = config.training
    if settings is None:
        raise ValueError("configuration needs a [training] section")
    output = Path(settings.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    records = load_records(settings.data)
    validation = load_records(settings.eval_data) if settings.eval_data else records
    mx.random.seed(settings.seed)
    engine = MLXDecisionEngine.load(config, trainable_adapter=True)
    if config.model.adapter_path:
        adapter_config = json.loads(
            (Path(config.model.adapter_path) / "adapter_config.json").read_text()
        )
    else:
        targets = config.lora.target_modules
        if isinstance(targets, str) and targets != "all-linear":
            raise ValueError("MLX target_modules must be all-linear or a list of module paths")
        adapter_config = {
            "fine_tune_type": "lora",
            "num_layers": config.lora.num_layers or len(engine.model.layers),
            "lora_parameters": {
                "rank": config.lora.rank,
                "scale": config.lora.alpha / config.lora.rank,
                "dropout": config.lora.dropout,
                **({"keys": targets} if isinstance(targets, list) else {}),
            },
        }
        install_lora(engine.model, adapter_config)
    encoded = [engine.encode(record.decision) for record in records]
    for record in validation:
        engine.encode(record.decision)
    if log:
        print(json.dumps({"phase": "validation_before", "records": len(validation)}), flush=True)
    before = evaluate_records(engine, validation)
    if settings.gradient_checkpointing:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(engine.model.layers[-1])
    optimizer = optim.AdamW(settings.learning_rate, weight_decay=settings.weight_decay)

    def objective(model, item, target):
        return loss_terms(candidate_logits(model, item), target)[0]

    value_and_grad = nn.value_and_grad(engine.model, objective)
    window_size = settings.batch_size * settings.gradient_accumulation_steps
    rng = random.Random(settings.seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format_version": 1,
        "backend": "mlx",
        "base_model": config.model.name_or_path,
        "base_revision": config.model.revision,
        "prompt": config.prompt.signature(),
        "labels": list(engine.codec.labels),
        "token_ids": list(engine.codec.token_ids),
        "objective": "forward_kl(target_action_distribution || model_action_distribution)",
        "normalization": "all_legal_actions",
    }
    write_json(output / "actlogit.json", manifest)
    write_json(output / "adapter_config.json", adapter_config)
    write_json(output / "training_config.json", config.model_dump())
    history = []
    start = time.perf_counter()
    engine.model.train()
    with (output / "training.jsonl").open("w") as events:
        for epoch in range(settings.epochs):
            indices = list(range(len(records)))
            rng.shuffle(indices)
            for offset in range(0, len(indices), window_size):
                window = indices[offset : offset + window_size]
                denominator = math.fsum(records[i].weight for i in window)
                grads, loss_sum = None, 0.0
                for index in window:
                    loss, grad = value_and_grad(
                        engine.model, encoded[index], mx.array(records[index].distribution())
                    )
                    weight = records[index].weight / denominator
                    grad = tree_map(lambda g, weight=weight: g * weight, grad)
                    grads = grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                    mx.eval(loss, grads)
                    loss_sum += loss.item() * weight
                grads, norm = optim.clip_grad_norm(grads, settings.max_grad_norm)
                if not math.isfinite(loss_sum) or not math.isfinite(norm.item()):
                    raise ValueError("nonfinite loss/gradient; check model precision and data")
                optimizer.update(engine.model, grads)
                mx.eval(engine.model.parameters(), optimizer.state)
                metrics = {
                    "step": len(history) + 1,
                    "epoch": epoch + 1,
                    "forward_kl": loss_sum,
                    "gradient_norm": norm.item(),
                    "seconds": time.perf_counter() - start,
                }
                history.append(metrics)
                events.write(json.dumps(metrics) + "\n")
                events.flush()
                if log:
                    print(json.dumps(metrics), flush=True)
                if len(history) % 25 == 0:
                    mx.save_safetensors(
                        str(output / "partial.safetensors"),
                        dict(tree_flatten(engine.model.trainable_parameters())),
                    )
                mx.clear_cache()
                if settings.max_steps and len(history) >= settings.max_steps:
                    break
            if settings.max_steps and len(history) >= settings.max_steps:
                break
    engine.model.eval()
    mx.save_safetensors(
        str(output / "adapters.safetensors"),
        dict(tree_flatten(engine.model.trainable_parameters())),
    )
    if log:
        print(json.dumps({"phase": "validation_after", "records": len(validation)}), flush=True)
    after = evaluate_records(engine, validation)
    report = {
        "backend": "mlx",
        "steps": len(history),
        "examples": len(records),
        "trainable_parameters": sum(
            v.size for _, v in tree_flatten(engine.model.trainable_parameters())
        ),
        "evaluation_split": "held_out" if settings.eval_data else "training",
        "before": before,
        "after": after,
        "history": history,
    }
    write_json(output / "metrics.json", report)
    return report
