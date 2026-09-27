# General capability evaluation

**English** | [简体中文](general-evaluation.zh-CN.md) · [Back to README](../README.md)

Compare a base model and its LoRA adapter on the same test questions. The built-in runner currently supports macOS + MLX. HumanEval code runs in a restricted subprocess using the macOS sandbox.

## Tasks

| Task | Questions | Metric |
|---|---:|---|
| MMLU | 250 | English multiple-choice accuracy across subjects |
| CMMLU | 250 | Chinese multiple-choice accuracy across subjects |
| GSM8K | 40 | Math final-answer accuracy |
| IFEval | 40 | Instruction following, strict / loose metrics |
| HumanEval | 20 | Python test pass rate, single-sample pass@1 |

The suite contains 600 questions, with pinned data versions, a fixed seed, and balanced subject sampling. Evaluation uses ordinary question-answering prompts rather than the decision API's prompt; generation tasks produce full text. This fixed subset supports before/after comparison, not official full-benchmark rankings.

## Run

Install the dependencies and copy the configuration template:

```bash
uv sync --extra mlx --extra server --extra eval
cp -n configs/mlx.example.toml configs/mlx.toml
```

`cp -n` preserves an existing configuration. The working `configs/mlx.toml` is ignored by Git. Set its model directory and keep the model and `[prompt]` settings consistent with adapter training.

```bash
# Download data and prepare the fixed suite on first use
uv run actlogit general-eval prepare

# Original base model
uv run actlogit general-eval run --config configs/mlx.toml \
  --base --output-dir outputs/general-eval/base

# The same base model with a domain adapter
uv run actlogit general-eval run --config configs/mlx.toml \
  --adapter outputs/domain-adapter --output-dir outputs/general-eval/trained

# Write Markdown and JSON comparison reports
uv run actlogit general-eval compare \
  --base outputs/general-eval/base --adapter outputs/general-eval/trained \
  --output outputs/general-eval/comparison.json
```

The helper script also runs preparation, both evaluations, and comparison:

```bash
ACTLOGIT_CONFIG=configs/mlx.toml \
ACTLOGIT_ADAPTER=outputs/domain-adapter \
bash scripts/general-eval.sh all
```

Use `bash scripts/general-eval.sh smoke` for a small pipeline check. It checks that the workflow runs, not model capability.

Runtime depends on the model, output lengths, and hardware. Run `--limit-per-task 5` in a separate output directory to estimate speed. Do not lower generation limits for a full evaluation.

## Resume and inspect results

Rerun the same command after an interruption. Use a new output directory after changing the model, adapter, configuration, or evaluation code. Comparison requires matching suites and evaluation protocols, with both runs complete.

| File | Contents |
|---|---|
| `base/results.jsonl`, `trained/results.jsonl` | Per-question responses, scores, timing, and truncation information |
| `summary.json` in each run directory | Task accuracy, IFEval metrics, and subject scores |
| `comparison.md` | Before/after scores, percentage-point changes, and paired confidence intervals |
| `comparison.json` | Structured report and IDs of questions that changed from correct to incorrect |

Inspect changes per task, then review regressions and truncation rates. Truncated answers remain in the statistics. No clear decline on a small sample does not prove that forgetting is absent. Investigate declines with larger evaluations of the affected tasks and weigh them against domain gains.

## Data sources

[MMLU](https://huggingface.co/datasets/cais/mmlu), [CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu), [GSM8K](https://huggingface.co/datasets/openai/gsm8k), [IFEval](https://huggingface.co/datasets/google/IFEval), and [HumanEval](https://huggingface.co/datasets/openai/openai_humaneval). Each dataset retains its own license.

See the [NOTICE](../src/actlogit/regression/_ifeval/NOTICE.md) for the IFEval checker's source and license. Keep all test questions separate from training data.
