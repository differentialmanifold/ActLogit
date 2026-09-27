# Fine-tuning guide

**English** | [简体中文](training.zh-CN.md) · [Back to README](../README.md)

ActLogit trains LoRA adapters from offline JSONL data. Choice, Noul, and Score can share a dataset, using the same question definitions as the API.

## Prepare data

One example per line:

```json
{"state":"The upload page crashes.","question":{"type":"choice","instructions":"Route this ticket.","criteria":{"billing":"Payments and refunds","technical":"Software issues"}},"target":{"billing":0.02,"technical":0.98}}
{"state":"Please refund my payment.","question":{"type":"noul","instructions":"Does the customer request a refund?","criteria":{"true":"Explicit request for money back","false":"No request for money back"}},"target":{"true":95,"false":5}}
{"state":"We cannot log in and have no workaround.","question":{"type":"score","instructions":"How severe is this issue?","criteria":["Cosmetic only","Feature impaired; workaround available","Blocking; no workaround"]},"target":{"0":0,"1":0.05,"2":0.95}}
```

- `state`: Text or JSON data to evaluate.
- `question`: One question in the same format as `/v1/systemone`; `type` is required.
- `target`: Nonnegative weights for every outcome, normalized automatically. Their sum must be positive.
- `weight`: Optional positive example weight; defaults to 1.
- `metadata`: Optional provenance or other information; excluded from the model input.

Choice targets use option IDs. Noul uses `"true"` and `"false"`. Score uses level indices starting at `"0"`. Target dictionary order does not affect alignment; the order of Score `criteria` determines the indices.

Hard labels can use one-hot targets: yes for Noul is `{"true":1,"false":0}`, and level 2 on a three-level Score is `{"0":0,"1":0,"2":1}`. A single continuous score, such as 1.5, does not uniquely determine a level distribution; define a labeling rule to obtain target weights. Do not use raw negative rewards as probabilities.

Keep a separate validation set. Put highly related examples from the same user, document, or trajectory in the same split. Public general-capability test questions should remain outside the domain training data.

```bash
uv run actlogit validate-data --data data/train.jsonl
uv run actlogit validate-data --data data/eval.jsonl
```

## Configure training

Copy the templates for your backend. These commands preserve existing local configurations:

```bash
# Transformers
cp -n configs/local.example.toml configs/local.toml
cp -n configs/train.example.toml configs/train.toml

# MLX
cp -n configs/mlx.example.toml configs/mlx.toml
cp -n configs/train-mlx.example.toml configs/train-mlx.toml
```

Working `configs/*.toml` files are ignored by Git; only `.example.toml` templates are committed. Use `configs/train.toml` for Transformers or `configs/train-mlx.toml` for MLX. Set the model directory and data paths:

```toml
[lora]
rank = 16
alpha = 32
dropout = 0.0
target_modules = "all-linear"

[training]
data = "data/train.jsonl"
eval_data = "data/eval.jsonl"
output_dir = "outputs/domain-adapter"
epochs = 3
batch_size = 2
gradient_accumulation_steps = 4
learning_rate = 0.0001
seed = 42
```

These settings are a starting point; tune the learning rate and training duration against your validation set. Transformers supports `gradient_checkpointing = true` to reduce training memory. MLX supports `lora.num_layers` to train only the final N blocks; omit it to cover all blocks. Both backends freeze the base model and update only LoRA parameters.

```bash
uv run actlogit train --config configs/train.toml
uv run actlogit evaluate --config configs/local.toml --data data/eval.jsonl
uv run actlogit evaluate --config configs/local.toml \
  --adapter outputs/domain-adapter --data data/eval.jsonl
```

For MLX, use `configs/train-mlx.toml` and `configs/mlx.toml` instead. Evaluation reports Forward KL, cross-entropy, and agreement with the target distribution's most likely outcome. Lower KL means a closer match to the targets. Without `eval_data`, the report uses the training set and does not measure generalization.

The output directory contains the adapter, loading configuration, tokenizer, and `metrics.json`. Serve it with the same model and `[prompt]` settings:

```bash
uv run actlogit serve --config configs/local.toml --adapter outputs/domain-adapter
```

To continue training an existing adapter, pass its directory with `--adapter` and set a new `output_dir` in the training configuration. This starts a new optimizer rather than resuming an interrupted step. Transformers and MLX adapter formats are not interchangeable.

## Training objective

ActLogit minimizes `D_KL(q || p)`, where `q` is the offline target distribution and `p` is the model's distribution over the current outcomes. For fixed targets, this gives the same parameter gradients as soft-label cross-entropy. Base weights remain frozen while LoRA updates the adapter parameters.

Targets may come from human labels, expert votes, search statistics, or previously collected teacher probabilities. Once prepared, training requires no external model service.

After training, check both your domain validation set and [general capability regression](general-evaluation.md) to measure task gains and changes to existing capabilities.
