# 通用在线策略蒸馏

[English](training.md) · [返回 README](../README.zh-CN.md)

ActLogit 训练语言模型在当前状态下选择动作。训练方案不指定模型、任务、候选数量或专家算法。
应用提供环境和专家；ActLogit 负责学生决策、分布监督、历史回放、参数更新与恢复。
当前内置参数更新器使用 MLX 或 Transformers 的 LoRA；在线蒸馏协议与参数更新方式独立，
它不意味着已经支持量化基座全参数训练。

## 训练循环

1. 从配置的本地语言模型开始，不要求专家数据集或预热。也可显式加载已有 adapter。
2. 固定当前学生参数，让学生独立完成一局/一个任务，保存每次执行动作前的状态。
3. 专家批量标注学生访问的所有状态，包括失误和失败任务，返回各动作的非负权重。
4. 第一轮只使用新状态；后续混合新状态与历史学生状态。默认历史样本占比目标为 20%，
   历史不足时使用现有数量、不重复补齐。容量受限的 reservoir 回放池保留跨轮样本。
5. 打乱混合样本、更新学生，再开始下一轮；模型和优化器在轮间持续保留。
6. 用隔离的固定种子独立验证学生。专家不参与验证动作选择，验证数据不进入回放池。

这是结合学生轨迹采集和数据聚合的在线策略蒸馏。带回放时，并非每条训练状态都来自最新策略。
训练阶段默认按学生合法动作概率采样；验证阶段取合法动作 argmax。
每轮训练 `training.epochs` 遍，建议起点为 1；`training.max_steps` 可限制每轮更新次数。
`batch_size × gradient_accumulation_steps` 是完整更新窗口大小，尾部按实际样本权重归一化。
`max_episode_steps` 默认不设上限；设置后，达到上限会标记截断并结束这一轮，下轮重新 reset。

## 应用提供两个适配器

配置使用 `module:factory` 导入应用中的 Python 工厂；工厂接收对应 options 表作为关键字参数。
从应用根目录运行，或把应用模块安装到同一 Python 环境。工厂是可信的本地代码。
接口定义见 [`actlogit.online`](../src/actlogit/online.py)。

```python
from actlogit.online import Observation, Transition, TeacherTarget

# environment = make_environment(**environment_options)
# environment.reset(seed=...) -> Observation
# environment.step(choice_id) -> Transition
# environment.close()

# teacher = make_teacher(**teacher_options)
# teacher.identity -> 非空 JSON 对象，包含固定模型/规则/搜索版本
# teacher.evaluate_batch(list[Observation]) -> list[TeacherTarget]
# teacher.close()
```

`Observation`：

- `state`：学生可见的 JSON 状态。
- `question`：与公开 API 相同的 Choice、Noul 或 Score 定义。
- `legal_choices`：可选的合法动作 ID 列表，省略表示所有候选合法。
- `context`：可选的教师侧上下文，不进入学生提示词。

环境 reset 必须返回至少一个合法候选。终止状态由 `Transition(terminated=True)` 表示。
非终止 transition 必须包含下一条 observation；`truncated` 表示外部截断。
`metrics` 是应用定义的累计数值指标，例如成功率或总收益；验证逐项报告均值及截断局数。
Score 在在线环境中执行的是离散等级 ID，Noul 执行 `true`/`false`，由环境解释其含义。

教师输出示例（动作 ID 与数量均由任务决定）：

```python
TeacherTarget(
    weights={"approve": 70, "review": 30, "reject": 0},
    metadata={"source": "search", "simulations": 100},
)
```

`weights` 可以是策略概率、MCTS 根节点访问次数、专家投票或 one-hot。
键必须精确覆盖全部候选，包括零权重项；所有值有限且非负，总和为正，非法动作权重必须为 0。
ActLogit 自动归一化并按 ID 对齐，不依赖字典顺序，也不要求师生共享词表或网络架构。
返回顺序必须与输入状态顺序一致。模型不会看到教师分布、搜索统计或 provenance 元数据。

教师应固定版本并对相同输入可复现。搜索适配器可以用状态哈希和固定 seed 派生独立随机流，
不能读取真实环境未来的随机结果。`identity` 应包含权重内容哈希及影响标签的搜索参数。
更换教师或搜索配置时开启新实验；可以加载旧 adapter 初始化，但它不是原实验的断点恢复。

## 目标函数

教师权重归一化为 q，学生候选 token logits 归一化为 p，最小化 `KL(q || p)`。
固定标签下它与软交叉熵 `-Σ q(a) log p(a)` 的参数梯度相同，支持搜索产生的零访问权重。
p 覆盖提示中的全部候选，非法候选的目标为 0，因而也受到合法性监督；执行时再按合法集合
重新归一化。合法 mask 不自动加入模型输入。没有合法概率质量或非有限梯度时停止并报错。

第一版不需要奖励模型、价值头、PPO 优势估计或整局回报反向传播。
专家价值/Q 值可以作为 metadata 保留，但不参与当前训练目标。

## 启动、恢复与推理

复制 [`configs/online.example.toml`](../configs/online.example.toml)，填写本地模型路径、
适配器工厂与应用 options。在线配置不填写 `training.data` 或 `training.eval_data`。

```bash
python -m actlogit.cli train-online --config configs/online.toml
python -m actlogit.cli train-online --config configs/online.toml --resume
# 扩大总轮数；已完成的轮数包含在 40 内
python -m actlogit.cli train-online --config configs/online.toml --resume --rounds 40
# 新输出目录，从配置的初始模型重新开始
python -m actlogit.cli train-online --config configs/online.toml --output-dir outputs/online-new
# 自动读取该实验最新完整 checkpoint 的 adapter
python -m actlogit.cli serve --config configs/online.toml --adapter outputs/online
```

输出结构：

- `episodes/round-*.json`：学生轨迹，含原始概率、实际动作、终止/截断及应用指标。
- `episodes/round-*.jsonl`：专家标注，兼容原有 TrainingRecord 格式。
- `checkpoints/round-*/adapter/`：可独立加载的 adapter；第 0 轮为更新前状态。
- `checkpoints/round-*/learner/`：优化器、训练步数与后端/采样随机状态。
- `checkpoints/round-*/replay.jsonl`、`state.json`：回放池、训练统计和固定种子验证。
- `baseline.json`：更新前的独立验证；`latest.json`：最新完整轮次。

检查点在一轮结束后提交。中途打断会恢复上一完整轮，然后重新采集并训练未完成轮，
不会声称恢复到中途的某个 optimizer step。模型初始化随机状态也保存于第 0 轮。
运行期间同一个输出目录只允许一个训练器。恢复检查配置与教师版本，允许增加总轮数；
学习率、回放比例、教师、验证设置等变化应使用新实验目录。
最新 checkpoint 不自动等于效果最佳；请根据独立验证选择具体轮次再做一次新种子最终测试。

MLX 的冻结前缀快速路径、分块 gated-delta 与真实 microbatch 是可选执行优化，默认关闭。
分块路径有架构限制，需在支持的模型上验证后启用；加速不代表任务质量改善。
冻结层快速路径只适用于第一个可训练 block 之前的冻结前缀。

## 已有离线数据

已有专家数据也可使用 `train` 单独训练，无需实现环境。每行是
`state + question + target`，`target` 与上述 weights 使用相同的非负权重规则。
详见 [`examples/train.jsonl`](../examples/train.jsonl) 与现有 `train*.example.toml`。

```bash
python -m actlogit.cli validate-data --data data/train.jsonl
python -m actlogit.cli train --config configs/train.toml
```

离线 `train --adapter ...` 是从权重开始的新优化器；在线 `train-online --resume` 才恢复
完整训练状态。在线主流程不依赖离线数据。另可运行[通用能力回归](general-evaluation.zh-CN.md)。

算法依据：[DAgger](https://proceedings.mlr.press/v15/ross11a.html)、
[GKD](https://arxiv.org/abs/2306.13649)、[Expert Iteration](https://arxiv.org/abs/1705.08439)。
这些方法提供设计依据，具体任务收益应通过独立实验确认。
