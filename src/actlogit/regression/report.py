from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from actlogit.regression.suite import COUNTS, write_json


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["task"]].append(row)
    tasks = {}
    for task, items in sorted(groups.items()):
        seconds = sum(r["inference_seconds"] for r in items)
        tasks[task] = {
            "n": len(items),
            "accuracy": sum(r["score"] for r in items) / len(items),
            "truncated": sum(r["finish_reason"] == "length" for r in items),
            "inference_seconds": seconds,
            "mean_seconds": seconds / len(items),
            "generation_tokens": sum(r["generation_tokens"] for r in items),
        }
        if task == "ifeval":
            metric = tasks[task]
            metric["prompt_loose"] = sum(r["prompt_loose"] for r in items) / len(items)
            for variant in ("strict", "loose"):
                values = [v for r in items for v in r[f"instructions_{variant}"]]
                metric[f"instruction_{variant}"] = sum(values) / len(values)
        if task in {"mmlu", "cmmlu"}:
            subjects = defaultdict(list)
            for row in items:
                subjects[row["subject"]].append(row["score"])
            tasks[task]["subjects"] = {
                key: {"n": len(values), "accuracy": sum(values) / len(values)}
                for key, values in sorted(subjects.items())
            }
    return {"tasks": tasks, "inference_seconds": sum(r["inference_seconds"] for r in rows)}


def paired_stats(before, after):
    import numpy as np

    differences = np.array(after, dtype=float) - np.array(before, dtype=float)
    rng = np.random.default_rng(42)
    means = rng.choice(differences, size=(10000, len(before)), replace=True).mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975]).tolist()
    return {
        "delta_pp": float(differences.mean() * 100),
        "paired_bootstrap_95_ci_pp": [low * 100, high * 100],
        "right_to_wrong": sum(a == 1 and b == 0 for a, b in zip(before, after, strict=True)),
        "wrong_to_right": sum(a == 0 and b == 1 for a, b in zip(before, after, strict=True)),
        "both_right": sum(a == 1 and b == 1 for a, b in zip(before, after, strict=True)),
        "both_wrong": sum(a == 0 and b == 0 for a, b in zip(before, after, strict=True)),
    }


def compare(base_dir, adapter_dir, output):
    from actlogit.regression.runner import read_results, verify_results

    directories = [Path(base_dir), Path(adapter_dir)]
    manifests = [json.loads((path / "run.json").read_text()) for path in directories]
    base, adapted = manifests
    if base["adapter"] is not None or adapted["adapter"] is None:
        raise ValueError("compare requires a base run followed by an adapter run")
    for key in (
        "format_version",
        "suite_sha256",
        "base",
        "protocol",
        "runtime",
        "checker",
        "selection",
        "full_suite",
    ):
        if base[key] != adapted[key]:
            raise ValueError(f"cannot compare: {key} differs")
    results = [read_results(path / "results.jsonl") for path in directories]
    for rows, manifest in zip(results, manifests, strict=True):
        verify_results(rows, manifest)
        if len(rows) != len(manifest["selection"]):
            raise ValueError("run is incomplete; resume it before comparing")
    summaries = [summarize(rows) for rows in results]
    tasks = {}
    regressions, improvements = [], []
    for task in COUNTS:
        before = [r["score"] for r in results[0] if r["task"] == task]
        after = [r["score"] for r in results[1] if r["task"] == task]
        tasks[task] = {
            "base": summaries[0]["tasks"][task],
            "adapter": summaries[1]["tasks"][task],
            **paired_stats(before, after),
        }
    for left, right in zip(*results, strict=True):
        if left["score"] > right["score"]:
            regressions.append(left["id"])
        if left["score"] < right["score"]:
            improvements.append(left["id"])
    report = {
        "full_suite": base["full_suite"],
        "sample_count": len(results[0]),
        "tasks": tasks,
        "right_to_wrong_ids": regressions,
        "wrong_to_right_ids": improvements,
        "total_inference_hours": sum(s["inference_seconds"] for s in summaries) / 3600,
        "base_dir": str(directories[0].resolve()),
        "adapter_dir": str(directories[1].resolve()),
        "interpretation": "Fixed-subset regression, not official leaderboard scores or "
        "proof of no catastrophic forgetting. CIs describe this sample and do not "
        "correct for multiple comparisons. Truncations remain in the denominator.",
    }
    output = Path(output)
    if output.suffix != ".json":
        raise ValueError("comparison output must end in .json (also writes .md)")
    write_json(output, report)
    lines = [
        "# 通用能力回归对比",
        "",
        f"题数：{report['sample_count']}；完整推荐协议：{report['full_suite']}；"
        f"两轮推理耗时：{report['total_inference_hours']:.2f} 小时。",
        "",
        "| 任务 | 题数 | 基座 | LoRA | 差值 pp | 配对 95% CI（pp） | "
        "对→错 | 错→对 | 截断 基座/LoRA |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for task, stats in tasks.items():
        b, a = stats["base"], stats["adapter"]
        lo, hi = stats["paired_bootstrap_95_ci_pp"]
        lines.append(
            f"| {task} | {b['n']} | {b['accuracy']:.1%} | {a['accuracy']:.1%} | "
            f"{stats['delta_pp']:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
            f"{stats['right_to_wrong']} | {stats['wrong_to_right']} | "
            f"{b['truncated']}/{a['truncated']} |"
        )
    lines += [
        "",
        "IFEval 主指标是 prompt-level strict；其他指令指标与学科分数见 JSON。",
        "",
        "这是固定子集、零样本聊天协议的配对回归，不等同于官方完整榜单。"
        "截断题仍计入分母；明显下降需要扩大该项样本复核。置信区间未做多重比较校正，"
        "小样本或零差异的窄区间不代表能力等价，不能证明不存在灾难性遗忘。",
        "",
        "答对变答错：" + (", ".join(regressions) or "无"),
        "",
    ]
    output.with_suffix(".md").write_text("\n".join(lines))
    return report
