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
    config.prompt.labels = ["B", "A", "C", "D", "E"]
    with pytest.raises(ValueError, match="mismatch"):
        load_engine(config)
    config.prompt.labels = ["A", "B", "C", "D", "E"]
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


def tiny_qwen():
    from mlx_lm.models.qwen3_5 import Model, ModelArgs

    mx.random.seed(21)
    model = Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config={
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 8,
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
    install_lora(
        model, {"num_layers": 4, "lora_parameters": {"rank": 4, "scale": 2, "dropout": 0.0}}
    )
    # A trained adapter, not all-zero B matrices: exercise both A and B gradients.
    for name, value in tree_flatten(model.trainable_parameters()):
        if name.endswith("lora_b"):
            model.load_weights([(name, 0.03 * mx.random.normal(value.shape))], strict=False)
    mx.eval(model.parameters())
    return model


def test_qwen_right_padding_and_fast_prefix_preserve_logits_and_weighted_gradients():
    import mlx.nn as nn

    from actlogit.mlx_backend import (
        EncodedDecision,
        candidate_logits_batch,
        set_training_mode,
    )

    model = tiny_qwen()
    examples = [
        EncodedDecision([3, 7, 4, 10, 2], (8, 11, 19)),
        EncodedDecision([2, 4, 9], (3, 7, 12)),
        EncodedDecision([2, 1, 8, 9, 2, 3, 4], (4, 5, 6)),
    ]
    targets = mx.array([[0.2, 0.8, 0], [0, 1, 0], [0.1, 0.3, 0.6]])
    weights = mx.array([1 / 6, 2 / 6, 3 / 6])

    def serial(m):
        return sum(
            loss_terms(candidate_logits(m, e), targets[i])[0] * weights[i]
            for i, e in enumerate(examples)
        )

    model.train()
    expected_logits = mx.stack([candidate_logits(model, e) for e in examples])
    old_loss, old_grads = nn.value_and_grad(model, serial)(model)
    mx.eval(expected_logits, old_loss, old_grads)
    assert set_training_mode(model) == 4
    assert all(not layer.training for layer in model.layers[:4])
    assert all(layer.training for layer in model.layers[4:])
    logits = candidate_logits_batch(model, examples)
    loss, grads = nn.value_and_grad(
        model,
        lambda m: mx.sum(loss_terms(candidate_logits_batch(m, examples), targets)[0] * weights),
    )(model)
    mx.eval(logits, loss, grads)
    assert mx.allclose(logits, expected_logits, atol=2e-5, rtol=2e-4).item()
    assert loss.item() == pytest.approx(old_loss.item(), abs=2e-5)
    old_grads, grads = dict(tree_flatten(old_grads)), dict(tree_flatten(grads))
    assert old_grads.keys() == grads.keys()
    for name in grads:
        assert mx.allclose(grads[name], old_grads[name], atol=2e-5, rtol=2e-3).item(), name


def test_fast_prefix_stops_before_any_trainable_block_and_embedding():
    from actlogit.mlx_backend import set_training_mode

    model = tiny_qwen()
    model.layers[1].unfreeze()
    assert set_training_mode(model) == 1
    assert all(layer.training for layer in model.layers[1:])
    model.language_model.model.embed_tokens.unfreeze()
    assert set_training_mode(model) == 0
    assert all(layer.training for layer in model.layers)


def test_real_microbatch_training_matches_accumulation_including_weighted_tail(
    config,
    choice_question,
    tmp_path,
):
    config.model.backend, config.model.device, config.model.dtype = "mlx", "auto", "auto"
    config.lora = LoRAConfig(rank=4, alpha=8, num_layers=1)
    records = [
        TrainingRecord(
            state="refund " * (1 + i % 3),
            question=choice_question,
            target={"billing": 0.1, "technical": 0.8, "account": 0.1},
            weight=i + 1,
        )
        for i in range(9)
    ]
    data = tmp_path / "weighted.jsonl"
    data.write_text("".join(r.model_dump_json() + "\n" for r in records))
    outputs = []
    for batch, accumulation in [(1, 8), (4, 2)]:
        output = tmp_path / f"batch-{batch}"
        config.training = TrainingConfig(
            data=str(data),
            output_dir=str(output),
            epochs=1,
            batch_size=batch,
            mlx_batching=True,
            gradient_accumulation_steps=accumulation,
            seed=42,
        )
        report = train(config, log=False)
        assert report["steps"] == 2
        assert [r["examples"] for r in report["history"]] == [8, 1]
        assert sum(r["examples"] for r in report["history"]) == 9
        outputs.append(mx.load(str(output / "adapters.safetensors")))
    for name in outputs[0]:
        assert mx.allclose(outputs[0][name], outputs[1][name], atol=2e-6, rtol=2e-4).item()


def test_microbatches_keep_optimizer_windows_and_mixed_candidate_counts():
    from types import SimpleNamespace

    from actlogit.mlx_backend import EncodedDecision, microbatches

    items = [
        EncodedDecision([1] * length, tuple(range(count)))
        for length, count in [(3, 4), (5, 4), (4, 2), (6, 4), (3, 4)]
    ]
    window = [3, 0, 2, 1, 4]
    batches = list(microbatches(SimpleNamespace(model_type="qwen3_5"), window, items, 4))
    assert sorted(i for b in batches for i in b) == sorted(window)
    assert all(len({len(items[i].candidate_ids) for i in b}) == 1 for b in batches)
    assert batches == [[3, 0, 1], [2], [4]]


def test_chunked_qwen_model_gradients_and_inference_are_preserved():
    import mlx.nn as nn

    from actlogit.mlx_backend import EncodedDecision, set_training_mode
    from actlogit.mlx_delta import training_delta_context

    model = tiny_qwen()
    example = EncodedDecision(([3, 7, 4, 10, 2] * 14)[:67], (8, 11, 19))

    def objective(m):
        return loss_terms(candidate_logits(m, example), mx.array([0.2, 0.8, 0.0]))[0]

    model.train()
    old_loss, old_grad = nn.value_and_grad(model, objective)(model)
    mx.eval(old_loss, old_grad)
    for fast_prefix in (False, True):
        set_training_mode(model, fast_frozen_prefix=fast_prefix)
        with training_delta_context(model, True):
            loss, grad = nn.value_and_grad(model, objective)(model)
            mx.eval(loss, grad)
        assert abs(loss.item() - old_loss.item()) < 1e-5
        for (name, a), (_, b) in zip(tree_flatten(old_grad), tree_flatten(grad), strict=True):
            assert mx.allclose(a, b, atol=3e-5, rtol=3e-3).item(), name
    model.eval()
    before = candidate_logits(model, example)
    with training_delta_context(model, True):
        after = candidate_logits(model, example)
    assert mx.array_equal(before, after).item()
