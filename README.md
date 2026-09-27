# ActLogit

**English** | [简体中文](README.zh-CN.md)

**Run typed decisions with local LLMs. Fine-tune with offline distributions.**

ActLogit provides a Jev-style `POST /v1/systemone` API for local language models, with **Choice, Noul, and Score** questions in a single request. It supports Transformers (CUDA / MPS / CPU), MLX (Apple Silicon), and offline **Forward KL + LoRA** fine-tuning.

## Quick start

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), and local model weights. Run commands from the repository root:

```bash
uv sync --extra local --extra server
cp -n configs/local.example.toml configs/local.toml
```

Only `configs/*.example.toml` templates are committed. The working `configs/*.toml` files are ignored by Git, so you can customize them for your machine. The copy commands use `-n` to preserve existing configurations.

Edit `configs/local.toml` and point `model.name_or_path` to a local Transformers model directory:

```toml
[model]
name_or_path = "/absolute/path/to/model"
device = "auto"
dtype = "auto"
local_files_only = true
max_prompt_tokens = 4096

[prompt]
format = "auto"
enable_thinking = false
```

Start the server:

```bash
uv run actlogit serve --config configs/local.toml --port 8000
```

For MLX models on Apple Silicon, use the MLX dependencies and template:

```bash
uv sync --extra mlx --extra server
cp -n configs/mlx.example.toml configs/mlx.toml
```

Set the model directory in `configs/mlx.toml`, then start:

```bash
uv run actlogit serve --config configs/mlx.toml --port 8000
```

Use a text-generation model supported by the selected backend. Models are loaded locally by default. For Transformers, you can also set a Hugging Face model ID and `local_files_only = false`.

## API requests

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

Example request:

```json
{
  "model": "actlogit",
  "state": {"ticket": "I was charged twice. Please refund the duplicate today."},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "technical": "Software bugs and connectivity",
        "account": "Account access and profile changes"
      }
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Does the customer explicitly request a refund?"
    },
    "urgency": {
      "type": "score",
      "instructions": "How soon does the customer need a response?",
      "criteria": ["No deadline", "Within a week", "Today or sooner"]
    }
  }
}
```

Results appear under `answers`, using the same question IDs:

| Type | Response fields | Meaning |
|---|---|---|
| `choice` | `choice`, `probabilities`, `confidence` | Selected option and the probability of each option |
| `noul` | `noul` | Probability of yes, from 0 to 1; no separate confidence field |
| `score` | `score`, `legend`, `probabilities`, `confidence` | Probability-weighted level index, from 0 to N−1 for N levels; may be fractional |

Choice accepts 1–255 options with string, JSON, or `null` descriptions, subject to the model's label capacity. Score accepts 2–10 levels ordered from low to high. Noul optionally accepts `criteria: {"true": "What yes means", "false": "What no means"}`. Each request supports up to 32 independent questions.

ActLogit uses Jev's three request and response structures with your local model. Its `confidence` is the maximum candidate probability; it does not reproduce Jev's confidence calculation or represent calibrated accuracy. Omit `model`, or set it to `actlogit`, `default`, or the configured model name. Token counts in `usage` are currently `null`.

- API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: `GET /health`
- The server listens on localhost by default and has no built-in authentication.

Use the same request file without starting a server:

```bash
uv run actlogit predict --config configs/local.toml --input examples/systemone.json
```

## Fine-tune with offline data

Train with **Forward KL + LoRA** to match target probability distributions. Targets can come from human labels, expert votes, search statistics, or teacher models. Once the data is prepared, training requires no external model service.

Each JSONL line contains `state`, one `question` using the API's question definition, and its `target` distribution:

```json
{"state":"Please refund the duplicate charge.","question":{"type":"noul","instructions":"Does the customer request a refund?"},"target":{"true":0.95,"false":0.05}}
```

| Question type | Keys in `target` |
|---|---|
| Choice | Every option ID from `criteria` |
| Noul | `"true"` and `"false"` |
| Score | Level indices as strings: `"0"`, `"1"`, … |

Targets may be probabilities or nonnegative counts. Include every outcome, including zero-weight ones. Hard labels can use one-hot distributions. See [examples/train.jsonl](examples/train.jsonl) for all three types and the [fine-tuning guide](docs/training.md) for data requirements and settings. The example data demonstrates the format; replace it with your domain training set and a separate validation set.

Copy the training configuration:

```bash
cp -n configs/train.example.toml configs/train.toml
```

Edit the model path, `training.data`, `training.eval_data`, and `training.output_dir` in `configs/train.toml`, then run:

```bash
uv run actlogit validate-data --data examples/train.jsonl
uv run actlogit train --config configs/train.toml

# Compare the base model and adapter on your domain validation set
uv run actlogit evaluate --config configs/local.toml --data examples/eval.jsonl
uv run actlogit evaluate --config configs/local.toml \
  --adapter outputs/domain-adapter --data examples/eval.jsonl

# Serve the fine-tuned model
uv run actlogit serve --config configs/local.toml \
  --adapter outputs/domain-adapter --port 8000
```

For MLX fine-tuning, copy its template and set the model and data paths:

```bash
cp -n configs/train-mlx.example.toml configs/train-mlx.toml
```

Use `configs/train-mlx.toml` for training and `configs/mlx.toml` for evaluation and serving. The commands are otherwise the same. Keep the model path and `[prompt]` settings consistent between training and inference.

## General capability evaluation

Domain validation measures task performance. General capability regression compares knowledge, math, instruction following, and coding before and after fine-tuning. The built-in evaluator currently supports **macOS + MLX**, using 600 fixed test questions: MMLU 250, CMMLU 250, GSM8K 40, IFEval 40, and HumanEval 20.

```bash
uv sync --extra mlx --extra server --extra eval
cp -n configs/mlx.example.toml configs/mlx.toml
uv run actlogit general-eval prepare
uv run actlogit general-eval run --config configs/mlx.toml \
  --base --output-dir outputs/general-eval/base
uv run actlogit general-eval run --config configs/mlx.toml \
  --adapter outputs/domain-adapter --output-dir outputs/general-eval/trained
uv run actlogit general-eval compare \
  --base outputs/general-eval/base --adapter outputs/general-eval/trained \
  --output outputs/general-eval/comparison.json
```

Set the model directory in your local `configs/mlx.toml` before running. Initial preparation downloads public test data. Results include `comparison.md`, task scores, and regression IDs; rerun the same command to resume an interrupted evaluation. Runtime depends on the model and hardware. This is a fixed-subset comparison, not an official full benchmark or proof against forgetting. See the [evaluation guide](docs/general-evaluation.md).

## License

[MIT License](LICENSE). Model weights and datasets retain their own licenses. The API follows [TypeSafe Jev](https://docs.typesafe.ai/primitives). ActLogit is an independent project that runs your choice of local model.
