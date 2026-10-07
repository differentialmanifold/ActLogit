import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from actlogit.config import Config, LoRAConfig, ModelConfig, OnlineConfig, TrainingConfig
from actlogit.online import (
    Observation,
    TeacherTarget,
    Transition,
    collect_episode,
    label_episode,
    run,
)
from actlogit.schema import ChoiceQuestion, TrainingRecord
from actlogit.trainer import make_trainer, resolve_adapter


class RoutingEnvironment:
    """A non-game task with three stable action IDs and action-dependent states."""

    closed = 0

    def observation(self):
        return Observation(
            state={"ticket": self.ticket, "previous_team": self.previous},
            question=ChoiceQuestion(
                criteria={"billing": "refund", "technical": "bug", "account": "password"}
            ),
            legal_choices=["technical", "account"],
            context={"internal": "teacher-only"},
        )

    def reset(self, *, seed):
        self.ticket, self.previous, self.steps = f"refund {seed}", "none", 0
        return self.observation()

    def step(self, choice_id):
        assert choice_id in {"technical", "account"}
        self.steps += 1
        self.previous = choice_id
        return Transition(
            observation=self.observation() if self.steps < 3 else None,
            terminated=self.steps == 3,
            metrics={"resolved": float(self.steps)},
        )

    def close(self):
        RoutingEnvironment.closed += 1


class RoutingTeacher:
    identity = {"id": "routing-expert", "revision": "v1"}

    def evaluate_batch(self, observations):
        # Deliberately reversed keys and unnormalized counts.
        return [
            TeacherTarget(weights={"account": 1, "technical": 9, "billing": 0})
            for _ in observations
        ]

    def close(self):
        pass


def make_environment():
    return RoutingEnvironment()


def make_teacher():
    return RoutingTeacher()


class FakeLearner:
    fail = False
    received = []

    def __init__(self, config):
        self.step = 0
        self.engine = SimpleNamespace(
            predict=lambda d: SimpleNamespace(
                probabilities={"billing": 0.9, "technical": 0.07, "account": 0.03}
            )
        )

    def fit(self, records, *, emit=None):
        if self.fail:
            raise ValueError("injected update failure")
        FakeLearner.received.append(records)
        self.step += math.ceil(len(records) / 8)
        return [{"step": self.step}]

    def save_adapter(self, path):
        Path(path).mkdir(parents=True)

    def save_state(self, path):
        Path(path).mkdir(parents=True)
        (Path(path) / "fake.json").write_text(json.dumps({"step": self.step}))

    def load_state(self, path):
        self.step = json.loads((Path(path) / "fake.json").read_text())["step"]


def online_config(output, rounds=2):
    return Config(
        model=ModelConfig(name_or_path="unused"),
        training=TrainingConfig(
            output_dir=str(output), epochs=1, batch_size=1, gradient_accumulation_steps=8
        ),
        online=OnlineConfig(
            environment=f"{__name__}:make_environment",
            teacher=f"{__name__}:make_teacher",
            rounds=rounds,
            replay_capacity=6,
            replay_fraction=0.25,
            eval_every=0,
        ),
    )


def test_student_drives_every_step_and_expert_labels_pre_action_states():
    import random

    learner = FakeLearner(None)
    episode = collect_episode(
        learner.engine,
        make_environment,
        seed=50,
        rng=random.Random(0),
        sample=False,
        max_steps=None,
    )
    assert episode["steps"] == 3 and episode["terminated"] and not episode["truncated"]
    assert [v["action"] for v in episode["visits"]] == ["technical"] * 3
    assert [v["observation"]["state"]["previous_team"] for v in episode["visits"]] == [
        "none",
        "technical",
        "technical",
    ]
    records = label_episode(episode, RoutingTeacher(), batch_size=2, round_number=1)
    assert records[0].distribution() == [0, 0.9, 0.1]
    assert "internal" not in records[0].decision.model_dump_json()
    assert records[0].state["previous_team"] == "none"


def test_online_no_warmup_replay_and_checkpoint_resume(tmp_path):
    FakeLearner.received = []
    config = online_config(tmp_path / "run", rounds=1)
    run(config, log=False, trainer_factory=FakeLearner)
    assert len(FakeLearner.received[0]) == 3
    config.online.rounds = 2
    result = run(config, resume=True, log=False, trainer_factory=FakeLearner)
    assert result["updates"] == 2
    assert len(FakeLearner.received[1]) == 4  # all three new states plus one old state
    assert {r.metadata["round"] for r in FakeLearner.received[1]} == {1, 2}
    assert resolve_adapter(result["adapter"]).endswith("round-000002/adapter")
    with pytest.raises(ValueError, match="not empty"):
        run(config, log=False, trainer_factory=FakeLearner)
    config.training.learning_rate *= 2
    with pytest.raises(ValueError, match="differs"):
        run(config, resume=True, log=False, trainer_factory=FakeLearner)


def test_failed_round_does_not_advance_checkpoint(tmp_path):
    config = online_config(tmp_path / "failed", rounds=1)
    FakeLearner.fail = True
    try:
        with pytest.raises(ValueError, match="injected"):
            run(config, log=False, trainer_factory=FakeLearner)
    finally:
        FakeLearner.fail = False
    assert resolve_adapter(config.training.output_dir).endswith("round-000000/adapter")
    run(config, resume=True, log=False, trainer_factory=FakeLearner)
    assert resolve_adapter(config.training.output_dir).endswith("round-000001/adapter")


def test_resume_recovers_committed_checkpoint_with_stale_pointer(tmp_path):
    config = online_config(tmp_path / "stale", rounds=1)
    run(config, log=False, trainer_factory=FakeLearner)
    (Path(config.training.output_dir) / "latest.json").unlink()
    config.online.rounds = 2
    result = run(config, resume=True, log=False, trainer_factory=FakeLearner)
    assert result["updates"] == 2
    assert resolve_adapter(config.training.output_dir).endswith("round-000002/adapter")


@pytest.mark.parametrize(
    "weights",
    [
        {"technical": 1},  # missing IDs
        {"account": 0, "technical": 0, "billing": 0},
        {"account": 0, "technical": 0, "billing": 1},  # illegal
        {"account": 0, "technical": float("nan"), "billing": 0},
    ],
)
def test_invalid_teacher_data_rejected(weights):
    env = RoutingEnvironment()
    obs = env.reset(seed=1)
    teacher = SimpleNamespace(
        identity={"id": "bad"}, evaluate_batch=lambda obs: [{"weights": weights}]
    )
    episode = {"visits": [{"observation": obs.model_dump()}], "seed": 1, "steps": 1}
    with pytest.raises(ValueError):
        label_episode(episode, teacher, batch_size=1, round_number=1)


def test_validation_is_separate_and_caps_are_marked(tmp_path):
    config = online_config(tmp_path / "eval", rounds=1)
    config.online.eval_every, config.online.eval_episodes = 1, 2
    config.online.max_episode_steps = 2
    run(config, log=False, trainer_factory=FakeLearner)
    state = json.loads(
        (Path(config.training.output_dir) / "checkpoints/round-000001/state.json").read_text()
    )
    assert state["fresh_records"] == 2
    assert state["episode"]["truncated"]
    assert state["validation"]["truncated_episodes"] == 2
    assert state["validation"]["mean"]["resolved"] == 2
    assert all(e["seed"] != state["episode"]["seed"] for e in state["validation"]["episodes"])


@pytest.mark.parametrize("backend", ["transformers", "mlx"])
def test_real_optimizer_resume_matches_uninterrupted_training(config, tmp_path, backend):
    if backend == "mlx":
        pytest.importorskip("mlx.core")
        config.model.backend, config.model.device, config.model.dtype = "mlx", "auto", "auto"
    config.lora = LoRAConfig(rank=4, alpha=8, dropout=0.1)
    config.training = TrainingConfig(
        output_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=0.01,
    )
    obs = RoutingEnvironment().reset(seed=1)
    record = TrainingRecord(
        state=obs.state,
        question=obs.question,
        target={"account": 0.1, "technical": 0.9, "billing": 0},
    )
    learner = make_trainer(config)
    learner.fit([record] * 3)
    learner.save_adapter(tmp_path / "adapter")
    learner.save_state(tmp_path / "state")
    second = learner.fit([record] * 3)
    expected = learner.engine.predict(record.decision).probabilities
    config.model.adapter_path = str(tmp_path / "adapter")
    restored = make_trainer(config)
    restored.load_state(tmp_path / "state")
    actual = restored.fit([record] * 3)
    assert restored.step == learner.step == 4
    assert [h["forward_kl"] for h in actual] == pytest.approx(
        [h["forward_kl"] for h in second], abs=2e-6
    )
    assert restored.engine.predict(record.decision).probabilities == pytest.approx(
        expected, abs=2e-6
    )
