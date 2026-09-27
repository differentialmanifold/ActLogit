from __future__ import annotations

import json
import sys
from collections import Counter

import pytest

from actlogit.regression.report import compare, paired_stats
from actlogit.regression.runner import read_results, verify_results
from actlogit.regression.scoring import execute_code, numerical_answer, sandbox_preflight, score
from actlogit.regression.suite import balanced_sample, digest, prompt_for, write_json


def test_subject_balancing_is_reproducible_and_input_order_independent():
    rows = [{"id": f"{s}/{i}", "subject": s} for s in "abc" for i in range(20)]
    picked = balanced_sample(rows, 10, 42)
    assert picked == balanced_sample(rows[::-1], 10, 42)
    assert picked != balanced_sample(rows, 10, 43)
    assert sorted(Counter(r["subject"] for r in picked).values()) == [3, 3, 4]
    assert len({r["id"] for r in picked}) == 10


def test_prompts_do_not_leak_reference_answers_or_tests():
    row = {"task": "mmlu", "question": "Q", "choices": ["one", "two", "three", "four"], "answer": 3}
    assert prompt_for(row) == prompt_for({**row, "answer": 0})
    assert "Q\nA. one" in prompt_for(row)
    row = {
        "task": "humaneval",
        "prompt": "def f():",
        "canonical_solution": "SECRET",
        "test": "SECRET_TEST",
    }
    assert "SECRET" not in prompt_for(row)
    assert prompt_for({"task": "ifeval", "prompt": "Keep ORIGINAL\n"}) == "Keep ORIGINAL\n"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Reasoning 99, answer 42", None),
        ("#### 1,200.00", "1.2E+3"),
        ("#### -3.50\n", "-3.5"),
        ("#### 3/4", None),
        ("#### 42 cats", None),
        ("#### 1\nRevision\n#### 2", "2"),
    ],
)
def test_numerical_extraction(text, expected):
    assert numerical_answer(text) == expected


def test_official_ifeval_strict_and_loose():
    pytest.importorskip("langdetect")
    pytest.importorskip("nltk")
    row = {
        "task": "ifeval",
        "key": 1,
        "prompt": "Reply with at least two placeholders.",
        "instruction_id_list": ["detectable_content:number_placeholders"],
        "kwargs": [{"num_placeholders": 2}],
    }
    assert score(row, "Dear [name], see you on [date].")["score"] == 1
    assert score(row, "Dear friend.")["score"] == 0
    assert score(row, "")["prompt_loose"] == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS sandbox")
def test_humaneval_correct_wrong_timeout_and_sandbox():
    pytest.importorskip("psutil")
    sandbox_preflight()
    row = {
        "task": "humaneval",
        "entry_point": "add",
        "test": "def check(candidate):\n    assert candidate(2, 3) == 5",
    }
    assert score(row, "```python\ndef add(a,b): return a+b\n```")["score"] == 1
    assert score(row, "def add(a,b): return a-b")["score"] == 0
    assert execute_code("while True: pass", timeout=0.2)["code_status"] == "timeout"
    assert execute_code("import os; os._exit(0)")["score"] == 0


def test_resume_repairs_only_interrupted_final_line(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_bytes(b'{"id":"a"}\n{"id":')
    assert read_results(path, repair_tail=True) == [{"id": "a"}]
    assert path.read_bytes() == b'{"id":"a"}\n'
    path.write_bytes(b'{"id":"a"}')
    assert read_results(path, repair_tail=True) == [{"id": "a"}]
    assert path.read_bytes().endswith(b"\n")
    path.write_bytes(b'bad\n{"id":"a"}\n')
    with pytest.raises(ValueError, match="invalid result"):
        read_results(path, repair_tail=True)


def test_resume_rejects_duplicate_order_and_identity(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"id":"a"}\n{"id":"a"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        read_results(path)
    manifest = {"selection": ["a", "b"]}
    with pytest.raises(ValueError, match="order"):
        verify_results([{"id": "b"}], manifest)
    with pytest.raises(ValueError, match="different run"):
        verify_results([{"id": "a", "run_sha256": "wrong"}], manifest)


def test_paired_report_uses_matching_complete_runs(tmp_path):
    pytest.importorskip("numpy")
    tasks = ["mmlu", "cmmlu", "gsm8k", "ifeval", "humaneval"]
    manifest = {
        "format_version": 1,
        "suite_sha256": "data",
        "base": "base",
        "adapter": None,
        "protocol": {},
        "runtime": {},
        "checker": {},
        "selection": tasks,
        "full_suite": False,
    }
    dirs = [tmp_path / "base", tmp_path / "adapter"]
    for index, path in enumerate(dirs):
        meta = {**manifest, "adapter": None if index == 0 else "weights"}
        write_json(path / "run.json", meta)
        rows = [
            {
                "id": t,
                "task": t,
                "subject": "test",
                "run_sha256": digest(meta),
                "score": 1 - index,
                "inference_seconds": 2,
                "generation_tokens": 3,
                "finish_reason": "length",
                "prompt_loose": 0,
                "instructions_strict": [False],
                "instructions_loose": [False],
            }
            for t in tasks
        ]
        (path / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = compare(*dirs, tmp_path / "comparison.json")
    assert result["right_to_wrong_ids"] == tasks
    assert result["tasks"]["mmlu"]["delta_pp"] == -100
    assert result["tasks"]["mmlu"]["base"]["truncated"] == 1
    assert (tmp_path / "comparison.md").exists()
    (dirs[1] / "results.jsonl").write_text("")
    with pytest.raises(ValueError, match="incomplete"):
        compare(*dirs, tmp_path / "comparison.json")
    write_json(dirs[1] / "run.json", {**manifest, "adapter": "w", "protocol": {"different": 1}})
    with pytest.raises(ValueError, match="protocol"):
        compare(*dirs, tmp_path / "comparison.json")


def test_paired_statistics_detect_regressions():
    pytest.importorskip("numpy")
    result = paired_stats([1, 1, 0, 0], [0, 1, 1, 0])
    assert result["delta_pp"] == 0
    assert result["right_to_wrong"] == result["wrong_to_right"] == 1
    assert result["both_right"] == result["both_wrong"] == 1
    assert result == paired_stats([1, 1, 0, 0], [0, 1, 1, 0])


def test_runner_resumes_after_failure_and_rejects_changed_protocol(tmp_path, monkeypatch):
    pytest.importorskip("mlx.core")
    from argparse import Namespace

    from actlogit.regression import runner

    tasks = ["cmmlu", "gsm8k", "humaneval", "ifeval", "mmlu"]
    suite = {
        "records": [{"id": t, "task": t} for t in tasks],
        "seed": 42,
        "sha256": "suite",
        "max_tokens": {t: 768 for t in tasks[1:4]},
    }
    config_path = tmp_path / "config.toml"
    config_path.write_text('[model]\nbackend="mlx"\nname_or_path="base"\n')
    monkeypatch.setattr(runner, "load_suite", lambda _: suite)
    monkeypatch.setattr(runner, "checkpoint_identity", lambda _: {"sha": "weights"})
    monkeypatch.setattr(runner, "runtime_identity", lambda: {})
    monkeypatch.setattr(runner, "checker_identity", lambda _: {})
    monkeypatch.setattr(runner, "sandbox_preflight", lambda: None)
    monkeypatch.setattr(
        runner,
        "score",
        lambda *_: {
            "score": 1,
            "prompt_loose": 1,
            "instructions_strict": [True],
            "instructions_loose": [True],
        },
    )
    calls, loads = [], []

    class FakeEvaluator:
        def __init__(self, config):
            assert config.model.adapter_path is None
            loads.append(True)

        def predict(self, record, max_tokens):
            if len(calls) == 2 and len(loads) == 1:
                raise ValueError("simulated interruption")
            calls.append(record["id"])
            return {
                "response": "answer",
                "finish_reason": "stop",
                "prompt_tokens": 1,
                "generation_tokens": 1,
                "inference_seconds": 0.1,
            }

    monkeypatch.setattr(runner, "MLXEvaluator", FakeEvaluator)
    args = Namespace(
        suite="suite.json",
        config=config_path,
        base=True,
        adapter=None,
        limit_per_task=None,
        max_new_tokens=None,
        nltk_data=tmp_path,
        output_dir=tmp_path / "run",
    )
    with pytest.raises(ValueError, match="simulated"):
        runner.run(args)
    assert len(read_results(args.output_dir / "results.jsonl")) == 2
    assert runner.run(args)["completed"] == 5
    assert calls == tasks
    assert runner.run(args)["completed"] == 5
    assert len(loads) == 2  # Completed reruns do not reload the model.
    args.max_new_tokens = 64
    with pytest.raises(ValueError, match="resume mismatch"):
        runner.run(args)
