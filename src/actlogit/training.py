from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import torch

from actlogit.config import Config
from actlogit.data import write_json
from actlogit.engine import DecisionEngine
from actlogit.loss import forward_kl
from actlogit.schema import TrainingRecord
from actlogit.trainer import tuples, write_manifest


def _targets(records: list[TrainingRecord], device: torch.device) -> torch.Tensor:
    width = max(len(record.decision.choices) for record in records)
    targets = torch.zeros((len(records), width), dtype=torch.float32, device=device)
    for index, record in enumerate(records):
        values = record.distribution()
        targets[index, : len(values)] = torch.tensor(values, device=device)
    return targets


@torch.inference_mode()
def evaluate_records(
    engine: DecisionEngine, records: list[TrainingRecord], batch_size: int = 4
) -> dict[str, float]:
    if engine.config.model.backend == "mlx":
        from actlogit.mlx_backend import evaluate_records as mlx_evaluate

        return mlx_evaluate(engine, records, batch_size)
    if not records or batch_size < 1:
        raise ValueError("evaluation needs records and a positive batch size")
    was_training = engine.model.training
    engine.model.eval()
    totals = {"forward_kl": 0.0, "cross_entropy": 0.0, "target_entropy": 0.0, "top1_agreement": 0.0}
    total_weight = math.fsum(record.weight for record in records)
    try:
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            logits, mask = engine.logits([engine.encode(record.decision) for record in batch])
            targets = _targets(batch, engine.device)
            losses = forward_kl(logits, targets, mask)
            weights = torch.tensor([r.weight for r in batch], device=engine.device)
            for key in ("forward_kl", "cross_entropy", "target_entropy"):
                totals[key] += float((getattr(losses, key) * weights).sum())
            prediction = logits.masked_fill(~mask, -torch.inf).argmax(dim=-1)
            chosen_target = targets.gather(1, prediction[:, None]).squeeze(1)
            # Accept any maximizer if the target has a tie.
            agreement = torch.isclose(chosen_target, targets.max(dim=-1).values).float()
            totals["top1_agreement"] += float((agreement * weights).sum())
    finally:
        engine.model.train(was_training)
    return {key: value / total_weight for key, value in totals.items()}


class TorchTrainer:
    """Persistent Transformers/PEFT learner shared by file and online training."""

    def __init__(self, config):
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import set_seed

        if config.lora.num_layers is not None:
            raise ValueError("lora.num_layers is currently supported only by the MLX backend")
        self.config, self.settings = config, config.training
        settings = self.settings
        set_seed(settings.seed)
        self.rng, self.step = random.Random(settings.seed), 0
        self.engine = DecisionEngine.load(config, trainable_adapter=True)
        if not config.model.adapter_path:
            self.engine.model = get_peft_model(
                self.engine.model,
                LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    lora_dropout=config.lora.dropout,
                    target_modules=config.lora.target_modules,
                    bias="none",
                ),
            )
        if settings.gradient_checkpointing:
            self.engine.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.engine.model.enable_input_require_grads()
        self.trainable = [p for p in self.engine.model.parameters() if p.requires_grad]
        if not self.trainable:
            raise ValueError("model has no trainable parameters")
        self.optimizer = torch.optim.AdamW(
            self.trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.engine.device.type == "cuda"
            and self.engine.model.get_input_embeddings().weight.dtype == torch.float16,
        )
        self.execution = {
            "microbatch_size": settings.batch_size,
            "effective_batch_size": settings.batch_size * settings.gradient_accumulation_steps,
        }
        self.engine.model.eval()

    @property
    def trainable_parameters(self):
        return sum(p.numel() for p in self.trainable)

    def fit(self, records, *, emit=None):
        if not records:
            raise ValueError("training needs at least one record")
        settings, engine = self.settings, self.engine
        encoded = [engine.encode(r.decision) for r in records]
        window_size = settings.batch_size * settings.gradient_accumulation_steps
        history = []
        engine.model.train()
        try:
            for epoch in range(settings.epochs):
                indices = list(range(len(records)))
                self.rng.shuffle(indices)
                for offset in range(0, len(indices), window_size):
                    started = time.perf_counter()
                    window = indices[offset : offset + window_size]
                    denominator = math.fsum(records[i].weight for i in window)
                    self.optimizer.zero_grad(set_to_none=True)
                    window_kl = 0.0
                    for start in range(0, len(window), settings.batch_size):
                        micro = window[start : start + settings.batch_size]
                        batch = [records[i] for i in micro]
                        logits, mask = engine.logits([encoded[i] for i in micro])
                        losses = forward_kl(logits, _targets(batch, engine.device), mask)
                        weights = torch.tensor([r.weight for r in batch], device=engine.device)
                        loss = (losses.forward_kl * weights).sum() / denominator
                        if not torch.isfinite(loss):
                            raise ValueError(
                                "nonfinite training loss; check model precision and data"
                            )
                        self.scaler.scale(loss).backward()
                        window_kl += float(loss.detach())
                    self.scaler.unscale_(self.optimizer)
                    norm = torch.nn.utils.clip_grad_norm_(
                        self.trainable, settings.max_grad_norm, error_if_nonfinite=True
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.step += 1
                    metrics = {
                        "step": self.step,
                        "epoch": epoch + 1,
                        "forward_kl": window_kl,
                        "examples": len(window),
                        "gradient_norm": float(norm),
                        "update_seconds": time.perf_counter() - started,
                    }
                    history.append(metrics)
                    if emit:
                        emit(metrics)
                    if settings.max_steps and len(history) >= settings.max_steps:
                        return history
        finally:
            engine.model.eval()
        return history

    def save_adapter(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.engine.model.save_pretrained(path, safe_serialization=True)
        self.engine.tokenizer.save_pretrained(path)
        write_manifest(path, self.engine)

    def save_state(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        state = {
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "rng_torch": torch.get_rng_state(),
        }
        if self.engine.device.type == "cuda":
            state["rng_cuda"] = torch.cuda.get_rng_state_all()
        if self.engine.device.type == "mps":
            state["rng_mps"] = torch.mps.get_rng_state()
        torch.save(state, path / "optimizer.pt")
        write_json(path / "trainer.json", {"step": self.step, "rng": self.rng.getstate()})

    def load_state(self, path):
        path = Path(path)
        state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng_torch"])
        if "rng_cuda" in state:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        if "rng_mps" in state:
            torch.mps.set_rng_state(state["rng_mps"])
        state = json.loads((path / "trainer.json").read_text())
        self.step = state["step"]
        self.rng.setstate(tuples(state["rng"]))


def train(config: Config, *, log=True):
    from actlogit.trainer import train_files

    return train_files(config, log=log)
