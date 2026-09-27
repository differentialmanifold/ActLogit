"""Lazy backend dispatch: MLX-only installations do not need PyTorch/PEFT."""

from __future__ import annotations

from actlogit.config import Config


def load_engine(config: Config, **kwargs):
    if config.model.backend == "mlx":
        from actlogit.mlx_backend import MLXDecisionEngine

        return MLXDecisionEngine.load(config, **kwargs)
    from actlogit.engine import DecisionEngine

    return DecisionEngine.load(config, **kwargs)


def train(config: Config, **kwargs):
    if config.model.backend == "mlx":
        from actlogit.mlx_backend import train as train_impl
    else:
        from actlogit.training import train as train_impl
    return train_impl(config, **kwargs)


def evaluate_records(engine, records, batch_size=4):
    if engine.config.model.backend == "mlx":
        from actlogit.mlx_backend import evaluate_records as evaluate_impl
    else:
        from actlogit.training import evaluate_records as evaluate_impl
    return evaluate_impl(engine, records, batch_size)
