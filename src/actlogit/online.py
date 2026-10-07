"""Task-independent, episode-based online policy distillation.

Applications supply environment and teacher factories. All actions are chosen by
the student; experts label visited states after collection. No warm-up data is used.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import random
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import Field, JsonValue, model_validator

from actlogit.data import load_records, write_json
from actlogit.schema import Question, StrictModel, TrainingRecord
from actlogit.trainer import make_trainer, tuples


class Observation(StrictModel):
    state: JsonValue
    question: Question
    legal_choices: list[str] | None = None
    context: dict[str, JsonValue] = Field(default_factory=dict)  # teacher only

    @property
    def decision(self):
        return self.question.to_decision(self.state)

    @property
    def legal_ids(self):
        return (
            self.legal_choices
            if self.legal_choices is not None
            else [c.id for c in self.decision.choices]
        )

    @model_validator(mode="after")
    def validate_legal(self):
        ids = {c.id for c in self.decision.choices}
        legal = self.legal_ids
        if not legal or len(set(legal)) != len(legal) or not set(legal) <= ids:
            raise ValueError("legal_choices must be a nonempty, unique subset of choice IDs")
        return self


class Transition(StrictModel):
    observation: Observation | None = None
    terminated: bool = False
    truncated: bool = False
    metrics: dict[str, float] = Field(default_factory=dict)  # cumulative task metrics

    @model_validator(mode="after")
    def next_state(self):
        if not (self.terminated or self.truncated) and self.observation is None:
            raise ValueError("a nonterminal transition needs its next observation")
        return self


class TeacherTarget(StrictModel):
    weights: dict[str, Annotated[float, Field(ge=0)]]  # probabilities, counts, or one-hot
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class Environment(Protocol):
    def reset(self, *, seed: int) -> Observation: ...
    def step(self, choice_id: str) -> Transition: ...
    def close(self) -> None: ...


class Teacher(Protocol):
    identity: dict[str, JsonValue]  # immutable model/search revision and options

    def evaluate_batch(self, observations: list[Observation]) -> list[TeacherTarget]: ...
    def close(self) -> None: ...


def factory(spec, options):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name:
        raise ValueError("plugin factory must use module:function syntax")
    return getattr(importlib.import_module(module), name)(**options)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, value)
    temporary.replace(path)


def write_records(path, records):
    with Path(path).open("w") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


@contextmanager
def run_lock(output):
    # OS locks are released after exceptions or a killed process; no stale PID file.
    with (output / ".lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt

                lock.seek(0)
                lock.write(b"0")
                lock.flush()
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise ValueError("another trainer is using this output directory") from exc
        yield


def collect_episode(engine, environment_factory, *, seed, rng, sample, max_steps, emit=None):
    env = environment_factory()
    visits, started = [], time.perf_counter()
    try:
        observation = Observation.model_validate(env.reset(seed=seed))
        while True:
            # Own an immutable snapshot even if the environment mutates its state.
            observation = observation.model_copy(deep=True)
            response = engine.predict(observation.decision)
            ids, probabilities = observation.legal_ids, response.probabilities
            expected = {c.id for c in observation.decision.choices}
            if set(probabilities) != expected or any(
                not math.isfinite(p) or p < 0 for p in probabilities.values()
            ):
                raise ValueError("student returned invalid choice probabilities")
            weights = [probabilities[key] for key in ids]
            if math.fsum(weights) <= 0:
                raise ValueError("student assigned no finite positive mass to legal actions")
            action = (
                rng.choices(ids, weights=weights)[0]
                if sample
                else max(ids, key=probabilities.__getitem__)
            )
            visits.append(
                {
                    "observation": observation.model_dump(),
                    "action": action,
                    "student_probabilities": probabilities,
                }
            )
            transition = Transition.model_validate(env.step(action))
            capped = max_steps is not None and len(visits) >= max_steps
            if emit and (len(visits) == 1 or len(visits) % 25 == 0):
                emit({"phase": "collect", "seed": seed, "steps": len(visits)})
            if transition.terminated or transition.truncated or capped:
                return {
                    "seed": seed,
                    "steps": len(visits),
                    "visits": visits,
                    "terminated": transition.terminated,
                    "truncated": transition.truncated or (capped and not transition.terminated),
                    "metrics": transition.metrics,
                    "seconds": time.perf_counter() - started,
                }
            observation = transition.observation
    finally:
        env.close()


def label_episode(episode, teacher, *, batch_size, round_number, emit=None):
    records = []
    for offset in range(0, len(episode["visits"]), batch_size):
        observations = [
            Observation.model_validate(v["observation"])
            for v in episode["visits"][offset : offset + batch_size]
        ]
        targets = teacher.evaluate_batch(observations)
        if len(targets) != len(observations):
            raise ValueError("teacher returned a different number of targets")
        for index, (observation, raw) in enumerate(zip(observations, targets, strict=True)):
            target = TeacherTarget.model_validate(raw)
            record = TrainingRecord(
                state=observation.state,
                question=observation.question,
                target=target.weights,
                metadata={
                    "teacher": teacher.identity,
                    "expert": target.metadata,
                    "round": round_number,
                    "seed": episode["seed"],
                    "step": offset + index,
                },
            )
            if any(w > 0 and key not in observation.legal_ids for key, w in record.target.items()):
                raise ValueError("teacher assigned positive weight to an illegal choice")
            records.append(record)
        if emit:
            emit({"phase": "label", "records": len(records), "total": episode["steps"]})
    return records


def mix_replay(fresh, replay, fraction, rng):
    count = min(len(replay), round(len(fresh) * fraction / (1 - fraction)))
    return [*fresh, *rng.sample(replay, count)]


def update_replay(replay, fresh, capacity, seen, rng):
    # Reservoir sampling bounds memory without letting only the last long game survive.
    for record in fresh:
        seen += 1
        if len(replay) < capacity:
            replay.append(record)
        elif capacity:
            index = rng.randrange(seen)
            if index < capacity:
                replay[index] = record
    return seen


def evaluate(engine, env_factory, settings, emit):
    episodes = [
        collect_episode(
            engine,
            env_factory,
            seed=settings.eval_seed + i,
            rng=random.Random(0),
            sample=False,
            max_steps=settings.max_episode_steps,
            emit=emit,
        )
        for i in range(settings.eval_episodes)
    ]
    keys = set.intersection(*(set(e["metrics"]) for e in episodes))
    return {
        "episodes": [{k: v for k, v in e.items() if k != "visits"} for e in episodes],
        "mean": {
            key: math.fsum(e["metrics"][key] for e in episodes) / len(episodes)
            for key in sorted(keys)
        },
        "truncated_episodes": sum(e["truncated"] for e in episodes),
    }


def checkpoint(output, trainer, state, replay):
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    final = checkpoints / f"round-{state['round']:06d}"
    if final.exists():
        raise ValueError(f"checkpoint already exists: {final}")
    temporary = Path(tempfile.mkdtemp(prefix="pending-", dir=checkpoints))
    trainer.save_adapter(temporary / "adapter")
    trainer.save_state(temporary / "learner")
    write_records(temporary / "replay.jsonl", replay)
    write_json(temporary / "state.json", state)
    temporary.rename(final)
    atomic_json(output / "latest.json", {"checkpoint": str(final.relative_to(output))})
    return final


def run(config, *, resume=False, log=True, trainer_factory=make_trainer):
    if config.training is None or config.online is None:
        raise ValueError("train-online needs [training] and [online]")
    if config.training.data or config.training.eval_data:
        raise ValueError("online training uses student episodes; omit training.data/eval_data")
    settings = config.online
    train_seeds = range(config.training.seed + 1, config.training.seed + settings.rounds + 1)
    if settings.eval_every and any(
        settings.eval_seed + i in train_seeds for i in range(settings.eval_episodes)
    ):
        raise ValueError("training and evaluation seeds overlap")
    output = Path(config.training.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output):
        return _run(config, output, resume, log, trainer_factory)


def _run(config, output, resume, log, trainer_factory):
    settings = config.online

    def emit(event):
        if log:
            print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)

    existing = [p for p in output.iterdir() if p.name != ".lock"]
    if existing and not resume:
        raise ValueError("output directory is not empty; use --resume or a new directory")
    completed = sorted((output / "checkpoints").glob("round-*/state.json"))
    if resume and not completed:
        raise ValueError("resume needs a completed checkpoint")
    teacher = factory(settings.teacher, settings.teacher_options)
    try:
        # Identity must describe the exact frozen teacher, not only its filename.
        identity = json.loads(json.dumps(teacher.identity, allow_nan=False))
        if not isinstance(identity, dict) or not identity:
            raise ValueError("teacher.identity must be a nonempty JSON object")
        signature_config = config.model_dump()
        signature_config["online"].pop("rounds")  # extending the total budget is supported
        signature = hashlib.sha256(
            json.dumps(
                {"config": signature_config, "teacher": identity},
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest()
        rng, replay, seen, current = random.Random(config.training.seed), [], 0, 0
        learner_config = config.model_copy(deep=True)
        if resume:
            # Recover a checkpoint committed just before a crash updating latest.json.
            saved = completed[-1].parent.resolve()
            if not saved.is_relative_to(output):
                raise ValueError("invalid checkpoint path")
            state = json.loads((saved / "state.json").read_text())
            if state["signature"] != signature:
                raise ValueError("resume configuration or teacher revision differs from checkpoint")
            if saved.name != f"round-{state['round']:06d}":
                raise ValueError("checkpoint round does not match its directory")
            current, seen = state["round"], state["seen"]
            if settings.rounds < current:
                raise ValueError("rounds is a total budget and cannot precede the saved round")
            rng.setstate(tuples(state["rng"]))
            replay = (
                load_records(saved / "replay.jsonl")
                if (saved / "replay.jsonl").stat().st_size
                else []
            )
            learner_config.model.adapter_path = str(saved / "adapter")
            atomic_json(output / "latest.json", {"checkpoint": str(saved.relative_to(output))})
        trainer = trainer_factory(learner_config)
        if resume:
            trainer.load_state(saved / "learner")
        else:
            checkpoint(
                output,
                trainer,
                {
                    "round": 0,
                    "seen": 0,
                    "rng": rng.getstate(),
                    "signature": signature,
                    "teacher": identity,
                    "updates": 0,
                },
                [],
            )
            write_json(output / "config.json", config.model_dump())

        def env_factory():
            return factory(settings.environment, settings.environment_options)

        if settings.eval_every and not (output / "baseline.json").exists():
            emit({"phase": "baseline"})
            atomic_json(
                output / "baseline.json", evaluate(trainer.engine, env_factory, settings, emit)
            )
        episodes_dir = output / "episodes"
        episodes_dir.mkdir(exist_ok=True)
        for number in range(current + 1, settings.rounds + 1):
            emit({"phase": "round_start", "round": number})
            episode = collect_episode(
                trainer.engine,
                env_factory,
                seed=config.training.seed + number,
                rng=rng,
                sample=settings.sample_actions,
                max_steps=settings.max_episode_steps,
                emit=emit,
            )
            atomic_json(episodes_dir / f"round-{number:06d}.json", episode)
            fresh = label_episode(
                episode,
                teacher,
                batch_size=settings.teacher_batch_size,
                round_number=number,
                emit=emit,
            )
            if teacher.identity != identity:
                raise ValueError("teacher identity changed during this run")
            write_records(episodes_dir / f"round-{number:06d}.jsonl", fresh)
            records = mix_replay(fresh, replay, settings.replay_fraction, rng)
            history = trainer.fit(records, emit=lambda event, n=number: emit({"round": n, **event}))
            seen = update_replay(replay, fresh, settings.replay_capacity, seen, rng)
            validation = None
            if settings.eval_every and (
                number % settings.eval_every == 0 or number == settings.rounds
            ):
                emit({"phase": "evaluation", "round": number})
                validation = evaluate(trainer.engine, env_factory, settings, emit)
            report = {
                "round": number,
                "signature": signature,
                "teacher": identity,
                "seen": seen,
                "rng": rng.getstate(),
                "updates": trainer.step,
                "fresh_records": len(fresh),
                "replayed_records": len(records) - len(fresh),
                "replay_size": len(replay),
                "history": history,
                "validation": validation,
                "episode": {k: v for k, v in episode.items() if k != "visits"},
            }
            checkpoint(output, trainer, report, replay)
            emit(
                {
                    "phase": "round_complete",
                    "round": number,
                    "updates": trainer.step,
                    "fresh_records": len(fresh),
                    "validation": validation,
                }
            )
        return {
            "rounds": settings.rounds,
            "updates": trainer.step,
            "output_dir": str(output),
            "adapter": str(output),
            "teacher": identity,
        }
    finally:
        teacher.close()
