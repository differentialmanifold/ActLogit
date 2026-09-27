"""Real MLX gradients, frozen base weights, reload, and weighted soft targets."""

# ruff: noqa: E402
import json

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx.utils import tree_flatten

from actlogit.backend import load_engine, train
from actlogit.config import LoRAConfig, TrainingConfig
from actlogit.mlx_backend import candidate_logits, install_lora, loss_terms
from actlogit.schema import NoulQuestion, ScoreQuestion, TrainingRecord


def test_loss_includes_zero_target_candidates():
    logits = mx.array([1.0, 2.0, 4.0])
    target = mx.array([0.25, 0.75, 0.0])
    kl, ce, entropy = loss_terms(logits, target)
    expected = -(0.25 * (1 - mx.logsumexp(logits)) + 0.75 * (2 - mx.logsumexp(logits)))
    assert ce.item() == pytest.approx(expected.item())
    assert kl.item() == pytest.approx((ce - entropy).item())
    grad = mx.grad(lambda value: loss_terms(value, target)[0])(logits)
    assert grad[2].item() > 0  # Zero teacher mass still contributes to the denominator.


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_real_mlx_training_reload_and_manifest(config, decision, choice_question, tmp_path, kind):
    config.model.backend = "mlx"
    config.model.device = "auto"
    config.model.dtype = "auto"
    config.lora = LoRAConfig(rank=4, alpha=8, num_layers=1)
    question, target, winner = {
        "choice": (
            choice_question,
            {"billing": 0.1, "technical": 0.85, "account": 0.05},
            "technical",
        ),
        "noul": (NoulQuestion(instructions="refund?"), {"true": 0.15, "false": 0.85}, "false"),
        "score": (
            ScoreQuestion(instructions="refund?", criteria=["refund", "bug", "password"]),
            {"0": 0.1, "1": 0.85, "2": 0.05},
            "1",
        ),
    }[kind]
    records = [TrainingRecord(state=decision.state, question=question, target=target)]
    decision = records[0].decision
    data = tmp_path / "train.jsonl"
    data.write_text(records[0].model_dump_json() + "\n")
    config.training = TrainingConfig(
        data=str(data),
        output_dir=str(tmp_path / "adapter"),
        epochs=20,
        batch_size=1,
        learning_rate=0.02,
        seed=12,
    )
    base = load_engine(config)
    base_probabilities = base.predict(decision).probabilities
    result = train(config, log=False)
    assert result["after"]["forward_kl"] < result["before"]["forward_kl"]
    assert result["steps"] == 20
    # The original base checkpoint is immutable; a fresh baseline is unchanged.
    assert load_engine(config).predict(decision).probabilities == pytest.approx(base_probabilities)
    config.model.adapter_path = config.training.output_dir
    reloaded = load_engine(config)
    response = reloaded.predict(decision)
    assert response.choice == winner
    assert set(response.probabilities) == set(target)
    config.prompt.enable_thinking = True
    with pytest.raises(ValueError, match="mismatch"):
        load_engine(config)
    config.prompt.enable_thinking = False
    manifest = tmp_path / "adapter" / "actlogit.json"
    value = json.loads(manifest.read_text())
    value["token_ids"][0] += 1
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="label mapping"):
        load_engine(config)


def test_only_lora_parameters_trainable(config, decision):
    import mlx.nn as nn
    import mlx.optimizers as optim

    config.model.backend = "mlx"
    config.model.device = "auto"
    config.model.dtype = "auto"
    engine = load_engine(config)
    nn.quantize(engine.model, group_size=32, bits=8)
    frozen = dict(tree_flatten(engine.model.parameters()))
    install_lora(
        engine.model,
        {
            "num_layers": 1,
            "lora_parameters": {
                "rank": 4,
                "scale": 2,
                "dropout": 0.0,
                "keys": ["self_attn.q_proj"],
            },
        },
    )
    assert all(
        name.endswith(("lora_a", "lora_b"))
        for name, _ in tree_flatten(engine.model.trainable_parameters())
    )
    encoded = engine.encode(decision)
    loss, gradients = nn.value_and_grad(
        engine.model,
        lambda m: loss_terms(candidate_logits(m, encoded), mx.array([0.0, 1.0, 0.0]))[0],
    )(engine.model)
    optim.Adam(0.01).update(engine.model, gradients)
    mx.eval(engine.model.parameters(), loss)
    updated = dict(tree_flatten(engine.model.parameters()))
    for key, value in frozen.items():
        name = key.replace("self_attn.q_proj.", "self_attn.q_proj.linear.")
        assert mx.array_equal(value, updated[name]).item()


def test_qwen_last_hidden_projection_matches_full_forward():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    from actlogit.mlx_backend import EncodedDecision

    model = Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 40,
                "linear_num_value_heads": 1,
                "linear_num_key_heads": 1,
                "linear_key_head_dim": 32,
                "linear_value_head_dim": 32,
            },
        )
    )
    model.eval()
    encoded = EncodedDecision([3, 7, 4, 10, 2], (8, 11, 19))
    expected = model(mx.array([encoded.input_ids]))[0, -1, mx.array(encoded.candidate_ids)]
    assert mx.allclose(candidate_logits(model, encoded), expected, atol=1e-5).item()


def test_missing_local_model_never_downloads(config):
    config.model.backend = "mlx"
    config.model.name_or_path = "does-not-exist/local-only"
    with pytest.raises(ValueError, match="existing local"):
        load_engine(config)


def test_real_mlx_model_through_http_worker(config, decision, choice_question):
    from fastapi.testclient import TestClient

    from actlogit.server import create_app

    config.model.backend = "mlx"
    config.model.device = "auto"
    config.model.dtype = "auto"
    engine = load_engine(config)
    expected = engine.predict(decision)
    response = TestClient(create_app(engine)).post(
        "/v1/systemone",
        json={"state": decision.state, "questions": {"team": choice_question.model_dump()}},
    )
    assert response.status_code == 200
    assert response.json()["answers"]["team"]["choice"] == expected.choice
    assert response.json()["answers"]["team"]["probabilities"] == pytest.approx(
        expected.probabilities
    )
