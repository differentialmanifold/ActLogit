# 微调指南

[English](training.md) | **简体中文** · [返回 README](../README.zh-CN.md)

ActLogit 使用离线 JSONL 数据训练 LoRA 适配器。Choice、Noul、Score 可混合在同一个数据集中，训练与接口使用相同的问题定义。

## 准备数据

每行一个样本：

```json
{"state":"The upload page crashes.","question":{"type":"choice","instructions":"Route this ticket.","criteria":{"billing":"Payments and refunds","technical":"Software issues"}},"target":{"billing":0.02,"technical":0.98}}
{"state":"Please refund my payment.","question":{"type":"noul","instructions":"Does the customer request a refund?","criteria":{"true":"Explicit request for money back","false":"No request for money back"}},"target":{"true":95,"false":5}}
{"state":"We cannot log in and have no workaround.","question":{"type":"score","instructions":"How severe is this issue?","criteria":["Cosmetic only","Feature impaired; workaround available","Blocking; no workaround"]},"target":{"0":0,"1":0.05,"2":0.95}}
```

- `state`：被判断的文本或 JSON 数据。
- `question`：与 `/v1/systemone` 中单个问题的格式一致，必须包含 `type`。
- `target`：全部候选的非负权重，自动归一化；总和必须大于零。
- `weight`：可选，样本权重，默认 1，必须大于零。
- `metadata`：可选，来源等辅助信息，不参与训练输入。

Choice 的目标键是选项 ID；Noul 是 `"true"` 和 `"false"`；Score 是从 `"0"` 开始的等级序号。目标字典顺序不影响对齐，Score 的 `criteria` 顺序决定等级序号。

硬标签可以转换为 one-hot：Noul 的“是”对应 `{"true":1,"false":0}`，三等级 Score 的等级 2 对应 `{"0":0,"1":0,"2":1}`。只有一个连续评分（例如 1.5）不能唯一确定等级分布；需要先制定标注规则，得到每个等级的目标权重。不要把原始负奖励直接当成概率。

保留独立验证集；同一用户、文档或轨迹的高度相关样本应放在同一数据划分中。公开通用测试题不应混入领域训练集。

```bash
uv run actlogit validate-data --data data/train.jsonl
uv run actlogit validate-data --data data/eval.jsonl
```

## 训练配置

先复制对应后端的配置模板。以下命令不会覆盖已有的本地配置：

```bash
# Transformers
cp -n configs/local.example.toml configs/local.toml
cp -n configs/train.example.toml configs/train.toml

# MLX
cp -n configs/mlx.example.toml configs/mlx.toml
cp -n configs/train-mlx.example.toml configs/train-mlx.toml
```

实际使用的 `configs/*.toml` 已被 Git 忽略，只有 `.example.toml` 模板会被提交。Transformers 使用 `configs/train.toml`；MLX 使用 `configs/train-mlx.toml`。填写模型目录和数据路径：

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

这些参数是起点，根据验证集调整学习率和训练轮数。Transformers 支持 `gradient_checkpointing = true` 节省训练显存。MLX 可设置 `lora.num_layers`，只训练末尾指定数量的 block；省略时覆盖全部 block。两种后端都冻结基座，仅更新 LoRA 参数。

```bash
uv run actlogit train --config configs/train.toml
uv run actlogit evaluate --config configs/local.toml --data data/eval.jsonl
uv run actlogit evaluate --config configs/local.toml \
  --adapter outputs/domain-adapter --data data/eval.jsonl
```

MLX 对应替换为 `configs/train-mlx.toml` 和 `configs/mlx.toml`。评估报告包括 Forward KL、交叉熵和目标最大概率选项命中率；KL 越低，预测分布越接近目标分布。未指定 `eval_data` 时报告使用训练集，不能据此判断泛化。

输出目录包含适配器、加载配置、tokenizer 和 `metrics.json`。推理时使用相同模型及 `[prompt]` 设置：

```bash
uv run actlogit serve --config configs/local.toml --adapter outputs/domain-adapter
```

继续训练已有适配器时，用 `--adapter` 指定原目录，并在训练配置中设置一个新的 `output_dir`。这会开始新的优化器，不是从中断步骤恢复。Transformers 和 MLX 的适配器格式不能互换。

## 训练目标

ActLogit 优化 `D_KL(q || p)`：`q` 是离线数据中的目标分布，`p` 是模型对本次候选的预测分布。目标固定时，这与软标签交叉熵产生相同的参数梯度。基座权重保持冻结，LoRA 更新适配器参数。

目标分布可来自人工标签、专家投票、搜索统计或预先收集的教师概率。准备好数据后，训练无需外部模型服务。

训练后同时检查领域验证集和[通用能力回归](general-evaluation.zh-CN.md)，分别确认任务收益与原有能力的变化。
