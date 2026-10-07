# Generic online policy distillation

**English** | [简体中文](training.zh-CN.md) · [README](../README.md)

Applications provide an environment and an expert. ActLogit trains a local language model
on states visited by that model, without requiring a warm-up dataset. The protocol specifies
no model family, domain, action count or expert algorithm. Built-in parameter updates currently
use MLX or Transformers LoRA; the algorithm is independent of this choice, but full-parameter
training of quantized weights is not implemented.

## Learning loop

1. Start from the configured local model, optionally with an explicit initial adapter.
2. Freeze the student version for one episode. The student chooses every executed action.
3. Ask the expert to label each pre-action state, including states from failed episodes.
4. Train on all new states mixed with a bounded reservoir of previous student states.
   The default replay fraction is 20%; use fewer old samples when the pool is too small.
   The first round uses only its newly collected states.
5. Keep the model and optimizer alive, update the student, then collect the next episode.
6. Periodically evaluate the student alone with separate fixed seeds and greedy legal actions.
   Evaluation states never enter training or replay.

This is online collection with dataset aggregation; replayed states are not strictly on-policy
for the newest student. Training collection samples from the student's legal action distribution.
Each round makes `training.epochs` passes (start with 1); `training.max_steps` optionally caps
updates per round. Full optimizer windows contain `batch_size * gradient_accumulation_steps`
examples; partial windows use their actual total weight.
`online.max_episode_steps` is unset by default. A cap marks an episode truncated and starts a
new episode next round; it does not silently report a completed task.

## Application integration

Factories use `module:function` imports and receive their options table as keyword arguments.
Run from the application root or install its module in the same Python environment.
Factories are trusted local application code. The protocols live in
[`actlogit.online`](../src/actlogit/online.py):

```python
from actlogit.online import Observation, Transition, TeacherTarget

# environment = make_environment(**environment_options)
# environment.reset(seed=...) -> Observation
# environment.step(choice_id) -> Transition
# environment.close()

# teacher = make_teacher(**teacher_options)
# teacher.identity -> nonempty JSON object with immutable expert provenance
# teacher.evaluate_batch(list[Observation]) -> list[TeacherTarget]
# teacher.close()
```

An `Observation` has student-visible `state` and `question` (the same Choice/Noul/Score
schema as the public API), optional `legal_choices` (stable IDs, all candidates by default),
and optional teacher-only `context`. Neither legality metadata nor teacher context is
injected into the student prompt. Reset must return a nonterminal observation with a legal
choice. A `Transition` has the next observation, `terminated`, `truncated`, and task-defined
cumulative numeric `metrics`. Nonterminal transitions require an observation. Evaluation
reports per-metric means and truncated-episode counts. Score executes a discrete level ID;
Noul executes `true` or `false`, interpreted by the application environment.

```python
TeacherTarget(
    weights={"approve": 70, "review": 30, "reject": 0},
    metadata={"source": "search", "simulations": 100},
)
```

Targets can be policy probabilities, MCTS root visit counts, votes or one-hot labels. Include
exactly every candidate ID, including zeros. Weights must be finite, nonnegative, with a positive
sum; illegal choices must have zero weight. ActLogit normalizes weights and aligns by ID,
independently of dictionary order, expert architecture or tokenizer. Return one target per input,
in order. Targets and provenance never enter the student prompt.

Experts should be fixed and reproducible for the same input. Include a weights hash and relevant
search parameters in `identity`. Search can derive private random streams from state hashes and
a fixed seed; it must not read the real environment's future randomness. A different expert or
search configuration starts a new experiment; loading the earlier adapter initializes weights
but does not constitute resuming the original experiment.

## Objective

Minimize `KL(q || p)`, where q is the normalized expert target and p is the student's candidate-token
softmax. For fixed targets this has the same gradient as soft-label cross-entropy. It accepts zero
search counts. The softmax covers all prompt candidates, including zero-target illegal choices,
so they also receive legality supervision. Execution renormalizes over legal actions.
Zero legal probability mass or nonfinite gradients cause an error.

The initial implementation requires no reward model, value head, PPO advantages or differentiation
through an episode. Optional expert values may be recorded in metadata but are not training targets.

## Run and resume

Copy [`configs/online.example.toml`](../configs/online.example.toml). Set the local model path,
application factories and their options. Omit `training.data` and `training.eval_data`.

```bash
python -m actlogit.cli train-online --config configs/online.toml
python -m actlogit.cli train-online --config configs/online.toml --resume
# Total budget, including already completed rounds
python -m actlogit.cli train-online --config configs/online.toml --resume --rounds 40
# Restart from the configured initial model in a new directory
python -m actlogit.cli train-online --config configs/online.toml --output-dir outputs/online-new
# Resolve the latest completed checkpoint automatically
python -m actlogit.cli serve --config configs/online.toml --adapter outputs/online
```

Artifacts:

- `episodes/round-*.json`: student trajectories, actions, probabilities and task metrics.
- `episodes/round-*.jsonl`: expert-labeled TrainingRecords.
- `checkpoints/round-*/adapter/`: independently loadable adapter; round zero precedes training.
- `checkpoints/round-*/learner/`: optimizer, step count and backend/training random state.
- `checkpoints/round-*/replay.jsonl` and `state.json`: replay, sampling RNG, metrics and validation.
- `baseline.json`: evaluation before updates; `latest.json`: latest complete checkpoint.

Checkpoints commit at round boundaries. Interrupted rounds restart from the last completed
round; this is not mid-update recovery. Initial learner state is checkpointed as round zero.
Only one trainer can write a run directory at a time. Resume verifies configuration and teacher
identity, allowing the total round budget to grow. Changes to learning rate, replay, teacher or
evaluation configuration require a new run. Latest does not mean best: select a round using
independent validation, then evaluate once on fresh test seeds.

MLX frozen-prefix inference kernels, chunked gated-delta and real microbatching remain optional
execution optimizations, disabled by default. Chunking is architecture-specific and must be
validated for the chosen model. Frozen-prefix acceleration only applies before the first trainable
block. Throughput improvements do not establish task quality.

## Existing offline labels

File training remains available for applications with existing expert data. Each JSONL line uses
`state + question + target`; target has the same nonnegative-weight rules as above. See
[`examples/train.jsonl`](../examples/train.jsonl) and `train*.example.toml` templates.

```bash
python -m actlogit.cli validate-data --data data/train.jsonl
python -m actlogit.cli train --config configs/train.toml
```

Offline `train --adapter ...` starts a new optimizer from saved weights. Online
`train-online --resume` restores training state. The online loop needs no offline dataset.
Also measure [general-capability regression](general-evaluation.md).

Background: [DAgger](https://proceedings.mlr.press/v15/ross11a.html),
[GKD](https://arxiv.org/abs/2306.13649), [Expert Iteration](https://arxiv.org/abs/1705.08439).
Their results motivate the design; gains on a new task require independent evaluation.
