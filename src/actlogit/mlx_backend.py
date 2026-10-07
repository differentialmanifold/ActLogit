"""Local MLX single-token decisions and weighted forward-KL LoRA training.

Microbatches share a model forward/backward. Qwen3.5 uses right padding and
gathers each sample's last real token; other models batch only equal lengths.
Only the frozen prefix before the first trainable block can use inference kernels.
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
from mlx.utils import tree_flatten, tree_map, tree_unflatten

from actlogit.config import Config, prompt_token_limit
from actlogit.data import write_json
from actlogit.mlx_delta import training_delta_context
from actlogit.prompt import TokenLabels, render_prompt
from actlogit.schema import DecisionRequest, DecisionResponse
from actlogit.trainer import resolve_adapter, tuples, write_manifest


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
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    logq = mx.log(mx.maximum(target, 1e-30))
    ce = -mx.sum(target * logp, axis=-1)
    entropy = -mx.sum(target * logq, axis=-1)
    return ce - entropy, ce, entropy


def candidate_logits_batch(model, encoded):
    """No left padding or shared recurrent cache; padding never precedes a real token."""
    if not encoded or any(not item.input_ids or not item.candidate_ids for item in encoded):
        raise ValueError("batch needs nonempty sequences and candidates")
    if len({len(item.candidate_ids) for item in encoded}) != 1:
        raise ValueError("a microbatch must have the same number of candidates")
    if len(encoded) == 1:
        return candidate_logits(model, encoded[0])[None, :]
    qwen = getattr(model, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}
    lengths = [len(item.input_ids) for item in encoded]
    if not qwen and len(set(lengths)) != 1:
        raise ValueError("unequal-length batching is supported only for Qwen3.5")
    width = max(lengths)
    # Qwen's attention and convolution/recurrent operations are causal. Gather BEFORE
    # right padding; no padded state is reused by another request or training batch.
    inputs = mx.array([item.input_ids + [0] * (width - len(item.input_ids)) for item in encoded])
    batch_indices = mx.arange(len(encoded))
    positions = mx.array(lengths) - 1
    if qwen:
        language = model.language_model
        hidden = language.model(inputs)[batch_indices, positions][:, None, :]
        if language.args.tie_word_embeddings:
            logits = language.model.embed_tokens.as_linear(hidden)
        else:
            logits = language.lm_head(hidden)
        logits = logits[:, 0, :]
    else:
        logits = model(inputs)[batch_indices, positions]
    candidates = mx.array([item.candidate_ids for item in encoded])
    return mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)


def set_training_mode(model, *, fast_frozen_prefix=True):
    """Keep gradients through every block at/after the FIRST trainable block."""
    model.train()
    if not fast_frozen_prefix or getattr(model, "model_type", None) not in {
        "qwen3_5",
        "qwen3_5_moe",
    }:
        return 0
    if tree_flatten(model.language_model.model.embed_tokens.trainable_parameters()):
        return 0  # Gradients to trainable input embeddings must traverse all blocks.
    count = 0
    for layer in model.layers:
        if tree_flatten(layer.trainable_parameters()):
            break
        layer.eval()
        count += 1
    return count


def microbatches(model, window, encoded, batch_size):
    """Partition a window without dropping, repeating or moving samples to another window."""
    qwen = getattr(model, "model_type", None) in {"qwen3_5", "qwen3_5_moe"}
    for start in range(0, len(window), batch_size):
        groups = {}
        for index in window[start : start + batch_size]:
            item = encoded[index]
            key = (len(item.candidate_ids), None if qwen else len(item.input_ids))
            groups.setdefault(key, []).append(index)
        yield from groups.values()


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

        settings = config.model.model_copy()
        if settings.adapter_path:
            settings.adapter_path = resolve_adapter(settings.adapter_path)
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


class MLXTrainer:
    """Persistent MLX learner; fit() never reloads the model or resets Adam."""

    def __init__(self, config):
        self.config, self.settings = config, config.training
        mx.random.seed(self.settings.seed)
        self.rng = random.Random(self.settings.seed)
        self.step = 0
        self.engine = MLXDecisionEngine.load(config, trainable_adapter=True)
        if config.model.adapter_path:
            self.adapter_config = json.loads(
                (Path(config.model.adapter_path) / "adapter_config.json").read_text()
            )
        else:
            targets = config.lora.target_modules
            if isinstance(targets, str) and targets != "all-linear":
                raise ValueError("MLX target_modules must be all-linear or module paths")
            self.adapter_config = {
                "fine_tune_type": "lora",
                "num_layers": config.lora.num_layers or len(self.engine.model.layers),
                "lora_parameters": {
                    "rank": config.lora.rank,
                    "scale": config.lora.alpha / config.lora.rank,
                    "dropout": config.lora.dropout,
                    **({"keys": targets} if isinstance(targets, list) else {}),
                },
            }
            install_lora(self.engine.model, self.adapter_config)
        if self.settings.gradient_checkpointing:
            from mlx_lm.tuner.trainer import grad_checkpoint

            grad_checkpoint(self.engine.model.layers[-1])
        self.optimizer = optim.AdamW(
            self.settings.learning_rate, weight_decay=self.settings.weight_decay
        )
        self.execution = {}
        self.engine.model.eval()

    @property
    def trainable_parameters(self):
        return sum(v.size for _, v in tree_flatten(self.engine.model.trainable_parameters()))

    def fit(self, records, *, emit=None):
        if not records:
            raise ValueError("training needs at least one record")
        settings, model = self.settings, self.engine.model
        encoded = [self.engine.encode(record.decision) for record in records]
        window_size = settings.batch_size * settings.gradient_accumulation_steps
        microbatch_size = settings.batch_size if settings.mlx_batching else 1
        frozen = set_training_mode(model, fast_frozen_prefix=settings.mlx_fast_frozen_prefix)
        self.execution = {
            "fast_frozen_prefix_layers": frozen,
            "microbatch_size": microbatch_size,
            "chunked_gated_delta": settings.mlx_chunked_gated_delta,
            "effective_batch_size": window_size,
            "clear_cache_interval": settings.mlx_clear_cache_interval,
        }

        def objective(m, items, target, weights):
            return mx.sum(loss_terms(candidate_logits_batch(m, items), target)[0] * weights)

        value_and_grad = nn.value_and_grad(model, objective)
        history, started = [], time.perf_counter()
        try:
            with training_delta_context(model, settings.mlx_chunked_gated_delta):
                for epoch in range(settings.epochs):
                    indices = list(range(len(records)))
                    self.rng.shuffle(indices)
                    for offset in range(0, len(indices), window_size):
                        update_started = time.perf_counter()
                        window = indices[offset : offset + window_size]
                        denominator = math.fsum(records[i].weight for i in window)
                        grads, loss_sum, batch_count = None, 0.0, 0
                        for micro in microbatches(model, window, encoded, microbatch_size):
                            loss, grad = value_and_grad(
                                model,
                                [encoded[i] for i in micro],
                                mx.array([records[i].distribution() for i in micro]),
                                mx.array([records[i].weight / denominator for i in micro]),
                            )
                            grads = (
                                grad if grads is None else tree_map(lambda a, b: a + b, grads, grad)
                            )
                            mx.eval(loss, grads)
                            loss_sum += loss.item()
                            batch_count += 1
                        grads, norm = optim.clip_grad_norm(grads, settings.max_grad_norm)
                        if not math.isfinite(loss_sum) or not math.isfinite(norm.item()):
                            raise ValueError(
                                "nonfinite loss/gradient; check model precision and data"
                            )
                        self.optimizer.update(model, grads)
                        mx.eval(model.parameters(), self.optimizer.state)
                        self.step += 1
                        metrics = {
                            "step": self.step,
                            "epoch": epoch + 1,
                            "forward_kl": loss_sum,
                            "gradient_norm": norm.item(),
                            "seconds": time.perf_counter() - started,
                            "update_seconds": time.perf_counter() - update_started,
                            "peak_memory_gib": mx.get_peak_memory() / 2**30,
                            "examples": len(window),
                            "microbatches": batch_count,
                        }
                        history.append(metrics)
                        if emit:
                            emit(metrics)
                        if (
                            settings.mlx_clear_cache_interval
                            and self.step % settings.mlx_clear_cache_interval == 0
                        ):
                            mx.clear_cache()
                        if settings.max_steps and len(history) >= settings.max_steps:
                            return history
        finally:
            model.eval()
        return history

    def save_adapter(self, path):
        path = Path(path)
        write_manifest(path, self.engine)
        write_json(path / "adapter_config.json", self.adapter_config)
        mx.save_safetensors(
            str(path / "adapters.safetensors"),
            dict(tree_flatten(self.engine.model.trainable_parameters())),
        )

    def save_state(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mx.eval(self.optimizer.state, mx.random.state[0])
        mx.savez(
            str(path / "optimizer.npz"),
            **dict(tree_flatten(self.optimizer.state)),
            rng=mx.random.state[0],
        )
        write_json(path / "trainer.json", {"step": self.step, "rng": self.rng.getstate()})

    def load_state(self, path):
        path = Path(path)
        arrays = mx.load(str(path / "optimizer.npz"))
        mx.random.state[0][:] = arrays.pop("rng")
        self.optimizer.state = tree_unflatten(list(arrays.items()))
        state = json.loads((path / "trainer.json").read_text())
        self.step = state["step"]
        self.rng.setstate(tuples(state["rng"]))


def train(config: Config, *, log=True):
    from actlogit.trainer import train_files

    return train_files(config, log=log)
