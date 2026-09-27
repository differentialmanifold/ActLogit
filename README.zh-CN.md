# ActLogit

[English](README.md) | **简体中文**

**用本地语言模型做类型化决策，用离线目标分布微调。**

ActLogit 为本地语言模型提供 Jev 风格的 `POST /v1/systemone` 接口，支持 **Choice、Noul、Score**，可在一次请求中组合多个问题。支持 Transformers（CUDA / MPS / CPU）和 MLX（Apple Silicon），以及基于离线数据的 **Forward KL + LoRA** 微调。

## 快速开始

需要 Python 3.11+、[uv](https://docs.astral.sh/uv/) 和本地模型权重。在仓库根目录执行：

```bash
uv sync --extra local --extra server
cp -n configs/local.example.toml configs/local.toml
```

仓库只提交 `configs/*.example.toml` 模板，实际使用的 `configs/*.toml` 已被 Git 忽略，可以按本机环境修改。复制命令使用 `-n`，不会覆盖已有配置。

编辑 `configs/local.toml`，将 `model.name_or_path` 改为本地 Transformers 模型目录：

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

启动服务：

```bash
uv run actlogit serve --config configs/local.toml --port 8000
```

Apple Silicon 使用 MLX 模型时，改用以下安装和配置模板：

```bash
uv sync --extra mlx --extra server
cp -n configs/mlx.example.toml configs/mlx.toml
```

在 `configs/mlx.toml` 中填写模型目录后启动：

```bash
uv run actlogit serve --config configs/mlx.toml --port 8000
```

模型需要是对应后端支持的文本生成模型。默认只读取本地权重；Transformers 也可填写 Hugging Face 模型 ID 并设置 `local_files_only = false`。

## 接口调用

```bash
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @examples/systemone.json
```

请求示例：

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

结果位于 `answers`，键名与请求中的问题 ID 相同：

| 类型 | 返回字段 | 含义 |
|---|---|---|
| `choice` | `choice`, `probabilities`, `confidence` | 选中的选项及各选项概率 |
| `noul` | `noul` | “是”的概率，范围 0–1；没有单独的 confidence |
| `score` | `score`, `legend`, `probabilities`, `confidence` | 等级序号的概率加权平均；N 个等级对应 0–N−1，可为小数 |

Choice 接受 1–255 个选项，描述可以是字符串、JSON 或 `null`；实际数量还受模型标签容量限制。Score 接受 2–10 个从低到高排列的等级。Noul 可选传入 `criteria: {"true": "是的条件", "false": "否的条件"}`。每次请求最多 32 个独立问题。

ActLogit 使用 Jev 的三种请求／响应结构，但运行你选择的本地模型；`confidence` 定义为最大候选概率，不复刻 Jev 的 confidence 算法，也不是校准后的正确率。`model` 可省略，或传入 `actlogit`、`default`、配置的模型名称。`usage` 中的 token 计数目前为 `null`。

- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：`GET /health`
- 服务默认监听本机地址，无内置鉴权。

不启动服务也可以使用相同的请求文件：

```bash
uv run actlogit predict --config configs/local.toml --input examples/systemone.json
```

## 离线数据微调

使用 **Forward KL + LoRA** 拟合目标概率分布。训练数据可来自人工标注、专家投票、搜索统计或教师模型；数据准备完成后，训练无需调用外部服务。

JSONL 每行包含 `state`、一个与接口定义相同的 `question`，以及该问题的目标分布 `target`：

```json
{"state":"Please refund the duplicate charge.","question":{"type":"noul","instructions":"Does the customer request a refund?"},"target":{"true":0.95,"false":0.05}}
```

| 问题类型 | `target` 的键 |
|---|---|
| Choice | `criteria` 中的所有选项 ID |
| Noul | `"true"`、`"false"` |
| Score | 等级序号字符串：`"0"`、`"1"`、… |

目标可使用概率或非负计数，所有候选都必须提供，包括零权重项。硬标签可写成 one-hot 分布。完整三类示例见 [examples/train.jsonl](examples/train.jsonl)，数据要求和参数见 [微调指南](docs/training.zh-CN.md)。示例数据仅用于跑通流程，请替换为领域训练集和独立验证集。

先复制训练配置模板：

```bash
cp -n configs/train.example.toml configs/train.toml
```

编辑 `configs/train.toml` 的模型路径、`training.data`、`training.eval_data` 和 `training.output_dir`，然后运行：

```bash
uv run actlogit validate-data --data examples/train.jsonl
uv run actlogit train --config configs/train.toml

# 比较领域验证集上的基座与适配器
uv run actlogit evaluate --config configs/local.toml --data examples/eval.jsonl
uv run actlogit evaluate --config configs/local.toml \
  --adapter outputs/domain-adapter --data examples/eval.jsonl

# 使用微调后的模型启动服务
uv run actlogit serve --config configs/local.toml \
  --adapter outputs/domain-adapter --port 8000
```

MLX 微调先复制对应模板，再填写模型和数据路径：

```bash
cp -n configs/train-mlx.example.toml configs/train-mlx.toml
```

训练使用 `configs/train-mlx.toml`，评估和启动服务使用 `configs/mlx.toml`；其余命令相同。保持训练和推理的模型路径及 `[prompt]` 配置一致。

## 通用能力验证

领域验证集用于检查任务效果；通用能力回归用于比较微调前后的知识、数学、指令遵循和代码能力。内置评测目前支持 **macOS + MLX**，包含固定的 600 道测试题：MMLU 250、CMMLU 250、GSM8K 40、IFEval 40、HumanEval 20。

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

运行前在本地 `configs/mlx.toml` 中设置模型目录。首次准备会下载公开测试数据。结果包括 `comparison.md`、各任务分数及退化题目；中断后可重跑原命令继续。运行时间取决于模型和硬件。这是固定子集的前后对比，不是官方完整榜单，也不能证明不存在遗忘。使用方式见 [通用能力评测](docs/general-evaluation.zh-CN.md)。

## 许可

[MIT License](LICENSE)。模型权重和数据集遵循各自许可证。接口参考 [TypeSafe Jev](https://docs.typesafe.ai/primitives)。ActLogit 是独立项目，运行你选择的本地模型。
