import json

import pytest
import torch
from transformers import AutoModelForCausalLM

from actlogit.config import LoRAConfig, TrainingConfig
from actlogit.engine import DecisionEngine
from actlogit.schema import NoulQuestion, ScoreQuestion, TrainingRecord
from actlogit.training import evaluate_records, train


@pytest.mark.parametrize("kind", ["choice", "noul", "score"])
def test_real_lora_training_reduces_kl_and_roundtrips(
    config, decision, choice_question, tmp_path, kind
):
    data = tmp_path / "train.jsonl"
    question, target = {
        "choice": (choice_question, {"billing": 0.85, "technical": 0.1, "account": 0.05}),
        "noul": (NoulQuestion(instructions="refund?"), {"false": 0.15, "true": 0.85}),
        "score": (
            ScoreQuestion(instructions="refund?", criteria=["refund", "bug", "password"]),
            {"2": 0.05, "0": 0.85, "1": 0.1},
        ),
    }[kind]
    record = TrainingRecord(state=decision.state, question=question, target=target)
    decision = record.decision
    # Five examples exercise a partial accumulation window (2*2 + 1).
    data.write_text("\n".join([record.model_dump_json()] * 5) + "\n")
    output = tmp_path / "adapter"
    config.lora = LoRAConfig(rank=4, alpha=8)
    config.training = TrainingConfig(
        data=str(data),
        output_dir=str(output),
        epochs=12,
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=0.01,
        max_steps=24,
    )
    report = train(config, log=False)
    assert report["steps"] == 24
    # A random frozen LM head has limited expressivity; verify a substantial loss reduction.
    assert report["after"]["forward_kl"] < report["before"]["forward_kl"] * 0.5
    assert (output / "adapter_model.safetensors").exists()
    assert not (output / "model.safetensors").exists()
    inference = config.model_copy(deep=True)
    inference.model.adapter_path = str(output)
    reloaded = DecisionEngine.load(inference)
    observed = evaluate_records(reloaded, [record])
    assert observed["forward_kl"] == pytest.approx(report["after"]["forward_kl"], abs=1e-5)
    again = DecisionEngine.load(inference)
    assert reloaded.predict(decision).probabilities == pytest.approx(
        again.predict(decision).probabilities, abs=1e-7
    )
    base = AutoModelForCausalLM.from_pretrained(config.model.name_or_path, local_files_only=True)
    for name, parameter in reloaded.model.get_base_model().named_parameters():
        if "lora_" not in name:
            original = base.get_parameter(name.replace(".base_layer.", "."))
            assert torch.equal(original, parameter), f"base weight changed: {name}"
    trainable = DecisionEngine.load(inference, trainable_adapter=True)
    assert all("lora_" in name for name, p in trainable.model.named_parameters() if p.requires_grad)
    with pytest.raises(ValueError, match="not empty"):
        train(config, log=False)

    manifest_path = output / "actlogit.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["token_ids"][0] = -1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="label mapping"):
        DecisionEngine.load(inference)
