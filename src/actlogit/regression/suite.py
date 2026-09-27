from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

COUNTS = {"mmlu": 250, "cmmlu": 250, "gsm8k": 40, "ifeval": 40, "humaneval": 20}
MAX_TOKENS = {"gsm8k": 768, "ifeval": 1536, "humaneval": 1024}
# Pin data independently of local Hugging Face cache contents.
SOURCES = {
    "mmlu": (
        "cais/mmlu",
        "c30699e8356da336a370243923dbaf21066bb9fe",
        "all/test-00000-of-00001.parquet",
    ),
    "cmmlu": ("haonan-li/cmmlu", "efcc940752ea4a1ea94d2727f11f83858d64fc8e", "cmmlu_v1_0_1.zip"),
    "gsm8k": (
        "openai/gsm8k",
        "740312add88f781978c0658806c59bc2815b9866",
        "main/test-00000-of-00001.parquet",
    ),
    "ifeval": (
        "google/IFEval",
        "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "ifeval_input_data.jsonl",
    ),
    "humaneval": (
        "openai/openai_humaneval",
        "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "openai_humaneval/test-00000-of-00001.parquet",
    ),
}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def balanced_sample(records, count, seed):
    """Equal allocation across subjects, with seeded remainder allocation."""
    if count > len(records) or count < 1:
        raise ValueError("sample size must be in 1..dataset size")
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["id"]):
        groups[record.get("subject", "all")].append(record)
    rng = random.Random(seed)
    subjects = sorted(groups)
    rng.shuffle(subjects)
    for group in groups.values():
        rng.shuffle(group)
    result = []
    while len(result) < count:
        for subject in subjects:
            if groups[subject] and len(result) < count:
                result.append(groups[subject].pop())
    return result


def read_source(task, path):
    if task == "cmmlu":
        rows = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if "test" not in Path(name).parts or not name.endswith(".csv"):
                    continue
                subject = Path(name).stem
                reader = csv.DictReader(io.StringIO(archive.read(name).decode("utf-8-sig")))
                for index, row in enumerate(reader):
                    rows.append(
                        {
                            "id": f"cmmlu/{subject}/{index}",
                            "task": task,
                            "subject": subject,
                            "question": row["Question"],
                            "choices": [row[label] for label in "ABCD"],
                            "answer": "ABCD".index(row["Answer"].strip()),
                        }
                    )
        return rows
    if task == "ifeval":
        return [
            {**json.loads(line), "id": f"ifeval/{json.loads(line)['key']}", "task": task}
            for line in Path(path).read_text().splitlines()
            if line.strip()
        ]
    import pyarrow.parquet as pq

    rows = pq.read_table(path).to_pylist()
    return [
        {**row, "id": f"{task}/{row.get('task_id', index)}", "task": task}
        for index, row in enumerate(rows)
    ]


def prepare(output, seed=42):
    from huggingface_hub import hf_hub_download

    output = Path(output)
    if output.exists():
        suite = load_suite(output)
        if suite["seed"] != seed:
            raise ValueError("existing suite uses a different seed; choose a new output file")
        return {"suite": str(output.resolve()), "records": len(suite["records"]), "reused": True}
    records, provenance = [], {}
    for task, count in COUNTS.items():
        repo, revision, filename = SOURCES[task]
        print(f"Preparing {task}: {count} records", flush=True)
        path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        population = read_source(task, path)
        selected = balanced_sample(population, count, seed)
        records.extend(selected)
        provenance[task] = {
            "repo": repo,
            "revision": revision,
            "file": filename,
            "sha256": file_digest(path),
            "population": len(population),
            "subjects": dict(Counter(r.get("subject", "all") for r in selected)),
        }
    # Separate tasks evenly during a run; small per-task pilots use the same first records.
    records.sort(key=lambda r: (r["task"], r["id"]))
    suite = {
        "format_version": 1,
        "name": "general-600-v1",
        "seed": seed,
        "counts": COUNTS,
        "max_tokens": MAX_TOKENS,
        "sources": provenance,
        "records": records,
    }
    suite["sha256"] = digest(suite)
    write_json(output, suite)
    return {"suite": str(output.resolve()), "records": len(records), "sha256": suite["sha256"]}


def load_suite(path):
    suite = json.loads(Path(path).read_text())
    content = {key: value for key, value in suite.items() if key != "sha256"}
    if suite.get("sha256") != digest(content) or suite.get("format_version") != 1:
        raise ValueError("suite checksum/version mismatch; prepare a new suite")
    records = suite["records"]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate suite IDs")
    if dict(Counter(r["task"] for r in records)) != COUNTS:
        raise ValueError("suite must contain the fixed 600-question task counts")
    return suite


def prompt_for(record):
    task = record["task"]
    if task in {"mmlu", "cmmlu"}:
        instruction = (
            "请选择正确答案，只输出一个字母（A、B、C 或 D）。"
            if task == "cmmlu"
            else "Choose the correct answer. Reply with one letter only: A, B, C, or D."
        )
        choices = "\n".join(
            f"{label}. {text}" for label, text in zip("ABCD", record["choices"], strict=True)
        )
        return f"{instruction}\n\n{record['question']}\n{choices}\n"
    if task == "gsm8k":
        return (
            record["question"] + "\n\nSolve step by step. End with '#### ' followed by "
            "only the final numerical answer."
        )
    if task == "ifeval":
        return record["prompt"]
    if task == "humaneval":
        return (
            "Complete the Python function below. Return the complete implementation, "
            "including its signature and any needed imports, in one Python code block. "
            "Do not include tests or usage examples.\n\n" + record["prompt"]
        )
    raise ValueError(f"unknown task: {task}")
