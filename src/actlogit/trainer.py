"""Shared file-training orchestration and checkpoint helpers.

Backend trainers keep their model, optimizer and random state alive across calls
to fit(). The online runner uses the same trainers without an offline dataset.
"""

from __future__ import annotations

import json
from pathlib import Path

from actlogit.data import load_records, write_json


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


def resolve_adapter(path):
    """Accept an adapter directory or a completed online run's root directory."""
    root = Path(path).expanduser()
    if (root / "latest.json").is_file():
        relative = json.loads((root / "latest.json").read_text())["checkpoint"]
        target = (root / relative).resolve()
        if not target.is_relative_to(root.resolve()) or not (target / "state.json").is_file():
            raise ValueError("invalid or incomplete online checkpoint")
        return str(target / "adapter")
    return str(root)


def make_trainer(config):
    if config.training is None:
        raise ValueError("configuration needs a [training] section")
    config = config.model_copy(deep=True)
    if config.model.adapter_path:
        config.model.adapter_path = resolve_adapter(config.model.adapter_path)
    if config.model.backend == "mlx":
        from actlogit.mlx_backend import MLXTrainer

        return MLXTrainer(config)
    from actlogit.training import TorchTrainer

    return TorchTrainer(config)


def write_manifest(path, engine):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(
        path / "actlogit.json",
        {
            "format_version": 1,
            "backend": engine.config.model.backend,
            "base_model": engine.config.model.name_or_path,
            "base_revision": engine.config.model.revision,
            "prompt": engine.config.prompt.signature(),
            "labels": list(engine.codec.labels),
            "token_ids": list(engine.codec.token_ids),
            "objective": "forward_kl(target_action_distribution || model_action_distribution)",
            "normalization": "candidate_actions",
        },
    )
    write_json(path / "training_config.json", engine.config.model_dump())


def train_files(config, *, log=True):
    from actlogit.backend import evaluate_records

    settings = config.training
    if settings is None or not settings.data:
        raise ValueError("file training requires training.data; use train-online for rollouts")
    output = Path(settings.output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    records = load_records(settings.data)
    validation = load_records(settings.eval_data) if settings.eval_data else records
    trainer = make_trainer(config)
    before = evaluate_records(trainer.engine, validation)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "training.jsonl").open("w") as events:

        def emit(metrics):
            events.write(json.dumps(metrics) + "\n")
            events.flush()
            if log:
                print(json.dumps(metrics), flush=True)

        history = trainer.fit(records, emit=emit)
    after = evaluate_records(trainer.engine, validation)
    trainer.save_adapter(output)
    report = {
        "backend": config.model.backend,
        "steps": trainer.step,
        "examples": len(records),
        "trainable_parameters": trainer.trainable_parameters,
        "evaluation_split": "held_out" if settings.eval_data else "training",
        "execution": trainer.execution,
        "before": before,
        "after": after,
        "history": history,
    }
    write_json(output / "metrics.json", report)
    return report
