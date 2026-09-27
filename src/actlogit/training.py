from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch

from actlogit.config import Config
from actlogit.data import load_records, write_json
from actlogit.engine import DecisionEngine, EncodedDecision
from actlogit.loss import forward_kl
from actlogit.schema import TrainingRecord


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


def train(config: Config, *, log: bool = True) -> dict[str, Any]:
    if config.model.backend == "mlx":
        from actlogit.mlx_backend import train as mlx_train

        return mlx_train(config, log=log)
    if config.lora.num_layers is not None:
        raise ValueError("lora.num_layers is currently supported only by the MLX backend")
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import set_seed

    settings = config.training
    if settings is None:
        raise ValueError("configuration needs a [training] section")
    output = Path(settings.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty; choose a new directory: {output}")
    records = load_records(settings.data)
    eval_records = load_records(settings.eval_data) if settings.eval_data else records
    set_seed(settings.seed)
    engine = DecisionEngine.load(config, trainable_adapter=True)
    if not config.model.adapter_path:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora.rank,
            lora_alpha=config.lora.alpha,
            lora_dropout=config.lora.dropout,
            target_modules=config.lora.target_modules,
            bias="none",
        )
        engine.model = get_peft_model(engine.model, peft_config)
    if settings.gradient_checkpointing:
        engine.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        engine.model.enable_input_require_grads()
    # Validate every prompt before making an optimizer update.
    encoded: list[EncodedDecision] = [engine.encode(record.decision) for record in records]
    for record in eval_records:
        engine.encode(record.decision)
    trainable = [parameter for parameter in engine.model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no trainable LoRA parameters")
    before = evaluate_records(engine, eval_records, settings.batch_size)
    optimizer = torch.optim.AdamW(
        trainable, lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=engine.device.type == "cuda"
        and engine.model.get_input_embeddings().weight.dtype == torch.float16,
    )
    rng = random.Random(settings.seed)
    step = 0
    history = []
    window_size = settings.batch_size * settings.gradient_accumulation_steps
    engine.model.train()
    for epoch in range(settings.epochs):
        indices = list(range(len(records)))
        rng.shuffle(indices)
        for offset in range(0, len(indices), window_size):
            window = indices[offset : offset + window_size]
            denominator = math.fsum(records[i].weight for i in window)
            optimizer.zero_grad(set_to_none=True)
            window_kl = 0.0
            for start in range(0, len(window), settings.batch_size):
                micro = window[start : start + settings.batch_size]
                batch = [records[i] for i in micro]
                logits, mask = engine.logits([encoded[i] for i in micro])
                losses = forward_kl(
                    logits,
                    _targets(batch, engine.device),
                    mask,
                )
                weights = torch.tensor([r.weight for r in batch], device=engine.device)
                loss = (losses.forward_kl * weights).sum() / denominator
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss; check model precision and data")
                scaler.scale(loss).backward()
                window_kl += float(loss.detach())
            # Dividing by the actual window weight also handles a partial final accumulation window.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, settings.max_grad_norm, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            step += 1
            metrics = {"step": step, "epoch": epoch + 1, "forward_kl": window_kl}
            history.append(metrics)
            if log:
                print(json.dumps(metrics), flush=True)
            if settings.max_steps is not None and step >= settings.max_steps:
                break
        if settings.max_steps is not None and step >= settings.max_steps:
            break
    after = evaluate_records(engine, eval_records, settings.batch_size)
    engine.model.eval()
    output.mkdir(parents=True, exist_ok=True)
    engine.model.save_pretrained(output, safe_serialization=True)
    engine.tokenizer.save_pretrained(output)
    manifest = {
        "format_version": 1,
        "base_model": config.model.name_or_path,
        "base_revision": config.model.revision,
        "prompt": config.prompt.model_dump(),
        "labels": list(engine.codec.labels),
        "token_ids": list(engine.codec.token_ids),
        "objective": "forward_kl(target_action_distribution || model_action_distribution)",
        "normalization": "all_legal_actions",
    }
    write_json(output / "actlogit.json", manifest)
    write_json(output / "training_config.json", config.model_dump())
    report = {
        "steps": step,
        "examples": len(records),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "evaluation_split": "held_out" if settings.eval_data else "training",
        "before": before,
        "after": after,
        "history": history,
    }
    write_json(output / "metrics.json", report)
    return report
