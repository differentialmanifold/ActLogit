from __future__ import annotations

import fcntl
import importlib.metadata
import json
import os
import platform
import time
from collections import Counter
from pathlib import Path

from actlogit.config import load_config
from actlogit.regression.scoring import checker_identity, sandbox_preflight, score
from actlogit.regression.suite import digest, file_digest, load_suite, prompt_for, write_json


def checkpoint_identity(directory):
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {directory}")
    files = [
        p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix in {".safetensors", ".json", ".model", ".txt", ".jinja"}
    ]
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError(f"no safetensors checkpoint in {directory}")
    return {"path": str(directory), "files": {p.name: file_digest(p) for p in files}}


def runtime_identity():
    root = Path(__file__).parent
    paths = sorted(root.glob("*.py")) + [root.parent / "mlx_backend.py"]
    return {
        "code": {p.name: file_digest(p) for p in paths},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers", "nltk", "langdetect", "numpy")
        },
    }


def read_results(path, *, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as handle:
                    handle.truncate(offset)
                break
            raise ValueError(f"invalid result line {index + 1}: {path}") from exc
        if not line.endswith(b"\n"):
            if not repair_tail:
                raise ValueError("result file has an unfinished final line; resume run first")
            with path.open("ab") as handle:
                handle.write(b"\n")
        rows.append(row)
        offset += len(line)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate result IDs")
    return rows


def verify_results(rows, manifest):
    expected = manifest["selection"]
    if [r["id"] for r in rows] != expected[: len(rows)]:
        raise ValueError("results do not match the selected suite/order")
    if any(row.get("run_sha256") != digest(manifest) for row in rows):
        raise ValueError("results belong to a different run")


class MLXEvaluator:
    def __init__(self, config):
        from actlogit.mlx_backend import MLXDecisionEngine

        self.engine = MLXDecisionEngine.load(config)

    def predict(self, record, max_tokens):
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        from actlogit.mlx_backend import EncodedDecision, candidate_logits

        engine = self.engine
        prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt_for(record)}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        tokens = engine.tokenizer.encode(prompt, add_special_tokens=False)
        if not tokens or len(tokens) > engine.config.model.max_prompt_tokens:
            raise ValueError(f"{record['id']}: {len(tokens)} prompt tokens exceed configuration")
        start = time.perf_counter()
        if record["task"] in {"mmlu", "cmmlu"}:
            ids = [engine.tokenizer.encode(c, add_special_tokens=False) for c in "ABCD"]
            if any(len(item) != 1 for item in ids):
                raise ValueError("general MCQ scoring requires single-token A/B/C/D labels")
            logits = candidate_logits(
                engine.model, EncodedDecision(tokens, tuple(item[0] for item in ids))
            )
            probabilities = mx.softmax(logits).tolist()
            response = "ABCD"[max(range(4), key=probabilities.__getitem__)]
            result = {
                "response": response,
                "probabilities": probabilities,
                "prompt_tokens": len(tokens),
                "generation_tokens": 0,
                "finish_reason": "choice",
            }
        else:
            parts, last = [], None
            for part in stream_generate(
                engine.model,
                engine.tokenizer,
                tokens,
                max_tokens=max_tokens,
                sampler=make_sampler(temp=0.0),
                prefill_step_size=256,
            ):
                parts.append(part.text)
                last = part
            if last is None:
                raise ValueError("MLX returned no generation result")
            result = {
                "response": "".join(parts),
                "prompt_tokens": last.prompt_tokens,
                "generation_tokens": last.generation_tokens,
                "finish_reason": last.finish_reason,
            }
        result["inference_seconds"] = time.perf_counter() - start
        mx.clear_cache()
        return result


def run(args):
    suite = load_suite(args.suite)
    config = load_config(args.config)
    if config.model.backend != "mlx":
        raise ValueError("general-eval currently supports the local MLX backend")
    config.model.adapter_path = (
        None if args.base else str(Path(args.adapter).expanduser().resolve())
    )
    if args.limit_per_task is not None and args.limit_per_task < 1:
        raise ValueError("limit-per-task must be positive")
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")
    selected, counts = [], Counter()
    for record in suite["records"]:
        if args.limit_per_task is None or counts[record["task"]] < args.limit_per_task:
            selected.append(record)
            counts[record["task"]] += 1
    caps = {
        task: min(cap, args.max_new_tokens) if args.max_new_tokens else cap
        for task, cap in suite["max_tokens"].items()
    }
    checker = checker_identity(args.nltk_data)
    sandbox_preflight()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another process is using this output directory") from exc
        print("Fingerprinting model and adapter files...", flush=True)
        manifest = {
            "format_version": 1,
            "suite_sha256": suite["sha256"],
            "base": checkpoint_identity(config.model.name_or_path),
            "adapter": checkpoint_identity(config.model.adapter_path)
            if config.model.adapter_path
            else None,
            "protocol": {
                "name": "general-600-zero-shot-chat-v1",
                "thinking": False,
                "temperature": 0,
                "max_tokens": caps,
                "max_prompt_tokens": config.model.max_prompt_tokens,
                "loader_prompt_config": config.prompt.model_dump(),
                "mcq": "single-token-label-argmax",
                "gsm8k": "strict-hash-numeric",
                "humaneval": "full-code-chat-greedy-pass-at-1-macos-sandbox",
            },
            "runtime": runtime_identity(),
            "checker": checker,
            "selection": [r["id"] for r in selected],
            "full_suite": len(selected) == 600 and caps == suite["max_tokens"],
        }
        metadata = output / "run.json"
        if metadata.exists():
            if json.loads(metadata.read_text()) != manifest:
                raise ValueError(
                    "resume mismatch: model/adapter/suite/protocol/runtime changed; "
                    "use a new output directory"
                )
        else:
            if (output / "results.jsonl").exists():
                raise ValueError("results exist without run.json; use a new output directory")
            write_json(metadata, manifest)
        result_file = output / "results.jsonl"
        results = read_results(result_file, repair_tail=True)
        verify_results(results, manifest)
        if len(results) < len(selected):
            print(f"Loading model; resume at {len(results)}/{len(selected)}", flush=True)
            evaluator = MLXEvaluator(config)
            import mlx.core as mx

            mx.random.seed(suite["seed"])
            with result_file.open("a") as handle:
                for record in selected[len(results) :]:
                    prediction = evaluator.predict(record, caps.get(record["task"], 1))
                    scored = score(record, prediction["response"])
                    row = {
                        "id": record["id"],
                        "task": record["task"],
                        "subject": record.get("subject"),
                        "run_sha256": digest(manifest),
                        **prediction,
                        **scored,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    results.append(row)
                    print(
                        f"[{len(results)}/{len(selected)}] {row['id']} "
                        f"score={row['score']} {row['inference_seconds']:.1f}s "
                        f"finish={row['finish_reason']}",
                        flush=True,
                    )
        from actlogit.regression.report import summarize

        summary = summarize(results)
        summary.update(
            completed=len(results), selected=len(selected), full_suite=manifest["full_suite"]
        )
        write_json(output / "summary.json", summary)
        return summary
