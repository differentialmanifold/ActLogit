import pytest
import torch

from actlogit.engine import DecisionEngine
from actlogit.prompt import TokenLabels


def test_batch_padding_matches_individual_inference(config, decision):
    engine = DecisionEngine.load(config)
    other = decision.model_copy(deep=True)
    other.state = {"ticket": "password " * 40}
    other.choices = other.choices[:2]
    batch = engine.predict_many([decision, other])
    for request, result in zip([decision, other], batch, strict=True):
        single = engine.predict(request)
        assert result.choice in {choice.id for choice in request.choices}
        assert sum(result.probabilities.values()) == pytest.approx(1, abs=1e-6)
        assert result.probabilities == pytest.approx(single.probabilities, abs=1e-6)
        assert result.confidence == max(result.probabilities.values())


def test_unknown_and_multi_token_labels_rejected(config):
    engine = DecisionEngine.load(config)
    for labels in (["not-in-vocabulary"], ["A B"], ["A", "A"], ["[EOS]"]):
        with pytest.raises(ValueError):
            TokenLabels.build(engine.tokenizer, labels)


def test_does_not_silently_truncate_or_drop_candidates(config, decision):
    engine = DecisionEngine.load(config)
    engine.config.model.max_prompt_tokens = 2
    with pytest.raises(ValueError, match="never silently truncated"):
        engine.predict(decision)
    with pytest.raises(ValueError, match="capacity"):
        engine.codec.for_count(6)


def test_no_autoregressive_generation_is_called(config, decision, monkeypatch):
    engine = DecisionEngine.load(config)

    def fail(*args, **kwargs):
        raise AssertionError("single-token decisions should read logits directly")

    monkeypatch.setattr(engine.model, "generate", fail)
    engine.predict(decision)


def test_candidate_logit_indices_are_exact(config, decision):
    engine = DecisionEngine.load(config)
    encoded = engine.encode(decision)
    logits, mask = engine.logits([encoded])
    with torch.inference_mode():
        output = engine.model(input_ids=torch.tensor([encoded.input_ids]), use_cache=False)
    assert mask.all()
    assert torch.allclose(logits[0], output.logits[0, -1, list(encoded.candidate_ids)], atol=1e-6)
