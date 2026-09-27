# 通用能力评测

[English](general-evaluation.md) | **简体中文** · [返回 README](../README.zh-CN.md)

比较基座和 LoRA 适配器在同一组测试题上的表现。内置运行器目前支持 macOS + MLX；HumanEval 代码在受限子进程中执行，需要 macOS sandbox。

## 测试内容

| 任务 | 题数 | 指标 |
|---|---:|---|
| MMLU | 250 | 英文多学科选择题准确率 |
| CMMLU | 250 | 中文多学科选择题准确率 |
| GSM8K | 40 | 数学题最终答案准确率 |
| IFEval | 40 | 指令遵循，strict / loose 指标 |
| HumanEval | 20 | Python 代码测试通过率，单次 pass@1 |

共 600 题。使用固定数据版本和随机种子，按学科均衡抽样。评测采用普通问答提示，不使用决策接口的提示；生成题会输出完整文本。该固定子集用于微调前后比较，不等同于各数据集的完整官方榜单。

## 运行

先安装依赖并复制配置模板：

```bash
uv sync --extra mlx --extra server --extra eval
cp -n configs/mlx.example.toml configs/mlx.toml
```

`cp -n` 保留已有配置。实际使用的 `configs/mlx.toml` 已被 Git 忽略；在其中设置模型目录，保持模型及 `[prompt]` 与适配器训练配置一致。

```bash
# 首次下载数据并生成固定题库
uv run actlogit general-eval prepare

# 原始基座
uv run actlogit general-eval run --config configs/mlx.toml \
  --base --output-dir outputs/general-eval/base

# 同一个基座加载领域适配器
uv run actlogit general-eval run --config configs/mlx.toml \
  --adapter outputs/domain-adapter --output-dir outputs/general-eval/trained

# 生成 Markdown 和 JSON 对比报告
uv run actlogit general-eval compare \
  --base outputs/general-eval/base --adapter outputs/general-eval/trained \
  --output outputs/general-eval/comparison.json
```

也可使用便捷脚本，一次执行准备、两轮评测和比较：

```bash
ACTLOGIT_CONFIG=configs/mlx.toml \
ACTLOGIT_ADAPTER=outputs/domain-adapter \
bash scripts/general-eval.sh all
```

小样本检查使用 `bash scripts/general-eval.sh smoke`。它只检查流程是否可用，不能用作能力结论。

运行时间取决于模型、输出长度和硬件。先用 `--limit-per-task 5` 跑一个独立目录，可估计当前机器的速度。正式评测不要设置更低的生成上限。

## 续跑与结果

中断后重跑原命令即可继续。更换模型、适配器、配置或评测代码后，请使用新输出目录；只有题库和评测协议一致且两轮均完成，才能生成对比。

| 文件 | 内容 |
|---|---|
| `base/results.jsonl`、`trained/results.jsonl` | 逐题回答、得分、耗时和截断信息 |
| 各目录 `summary.json` | 各任务准确率、IFEval 指标和学科分数 |
| `comparison.md` | 前后分数、变化百分点、配对置信区间 |
| `comparison.json` | 结构化报告及答对变答错的题目 ID |

先看各任务的变化，再检查退化题目和截断率。截断题仍参与统计。小样本的“未明显下降”不能证明不存在遗忘；发现下降后应扩大对应任务的评测样本，并结合领域收益决定是否使用适配器。

## 数据来源

[MMLU](https://huggingface.co/datasets/cais/mmlu)、[CMMLU](https://huggingface.co/datasets/haonan-li/cmmlu)、[GSM8K](https://huggingface.co/datasets/openai/gsm8k)、[IFEval](https://huggingface.co/datasets/google/IFEval)、[HumanEval](https://huggingface.co/datasets/openai/openai_humaneval)。数据集各自的许可证仍然适用。

IFEval 检查器的来源和许可证见 [NOTICE](../src/actlogit/regression/_ifeval/NOTICE.md)。所有测试题都应与训练数据分离。
