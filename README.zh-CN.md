# ActLogit

[English](README.md) | **简体中文**

**用本地语言模型做类型化决策，在学生自己产生的轨迹上向专家学习。**

ActLogit 为本地语言模型提供 Jev-compatible（接口兼容）的 `POST /v1/systemone` 接口，支持 **Choice、Noul、Score**，可在一次请求中组合多个问题。支持 Transformers（CUDA / MPS / CPU）和 MLX（Apple Silicon），以及由应用提供环境和专家的**在线策略蒸馏**；同时兼容已有离线标签。

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

模型必须放在已存在的本地目录中，并且是所选后端支持的文本生成模型。ActLogit 不负责下载模型权重。

ActLogit 优先使用分词器的聊天模板并关闭思考，没有模板时自动使用普通文本提示词，无需配置提示词格式。请选择支持直接回答或非思考模式的模型。

`model.max_prompt_tokens` 是可选项。省略时使用模型/分词器声明的上下文上限；也可设置较小的值（例如 `4096`）控制部署延迟和内存占用。实际限制取部署预算与模型上限的较小值，包含指令、state、候选项和模板 token。超长输入会明确拒绝，不会静默截断。模型未提供有效上下文信息时，需显式配置此项。

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

Choice 接受 1–255 个选项，与 [Jev Choice 的限制](https://docs.typesafe.ai/primitives/choice)一致，实际数量还受分词器有效单 token 标签容量限制。描述可以是字符串、JSON 或 `null`；空字符串、空白字符串或 `null` 会使用选项名称作为描述。Score 接受 2–10 个从低到高排列的等级。Noul 可选传入 `criteria: {"true": "是的条件", "false": "否的条件"}`。每次请求最多 32 个独立问题。

ActLogit 使用 Jev 的三种请求／响应结构，但运行你选择的本地模型；`confidence` 定义为最大候选概率，不复刻 Jev 的 confidence 算法，也不是校准后的正确率。`model` 可省略，或传入 `actlogit`、`default`、配置的模型名称。`usage` 中的 token 计数目前为 `null`。

- API 文档：[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 健康检查：`GET /health`
- 服务默认监听本机地址，无内置鉴权。

不启动服务也可以使用相同的请求文件：

```bash
uv run actlogit predict --config configs/local.toml --input examples/systemone.json
```

## 训练决策能力

直接从本地语言模型开始：学生独立完成一局/一个任务，专家标注学生访问的每个状态，
再混合少量历史学生状态更新模型，无需专家预热数据。专家可以提供概率、投票、搜索访问
次数或 one-hot 标签，通过稳定动作 ID 对齐；通用训练循环不绑定具体任务或模型。

应用通过 [`configs/online.example.toml`](configs/online.example.toml) 配置环境与专家工厂。
当前内置 MLX、Transformers 学习器使用 LoRA，并在轮间持续保留优化器。

```bash
uv run actlogit train-online --config configs/online.toml
uv run actlogit train-online --config configs/online.toml --resume
uv run actlogit serve --config configs/online.toml --adapter outputs/online
```

运行前填写模型路径、应用工厂及其参数。接口协议、回放、独立验证和完整断点恢复见
[通用训练方案](docs/training.zh-CN.md)。

已有 JSONL 标签可继续使用 `actlogit train --config configs/train.toml`，配置模板为
`train.example.toml` 或 `train-mlx.example.toml`。每条记录包含 `state`、一个 API `question`
及覆盖全部候选的 `target` 权重。格式示例见 [examples/train.jsonl](examples/train.jsonl)。

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
