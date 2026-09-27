from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

from actlogit.regression.suite import file_digest


def prepare_checker(cache):
    """Download NLTK's sentence tables explicitly, never during inference."""
    import nltk

    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if not nltk.download("punkt_tab", download_dir=str(cache), quiet=True, raise_on_error=True):
        raise ValueError("could not download NLTK punkt_tab")
    return checker_identity(cache)


def checker_identity(cache):
    import nltk
    from langdetect import DetectorFactory

    cache = Path(cache).resolve()
    # Do not silently fall back to another user's/global NLTK corpus.
    nltk.data.path[:] = [str(cache)]
    nltk.sent_tokenize("Check one. Check two.")
    DetectorFactory.seed = 0
    files = sorted((Path(__file__).parent / "_ifeval").glob("*.py"))
    identity = {p.name: file_digest(p) for p in files}
    for path in sorted((cache / "tokenizers/punkt_tab/english").glob("*.txt")):
        identity[path.name] = file_digest(path)
    for path in sorted((cache / "tokenizers/punkt_tab/english").glob("*.tab")):
        identity[path.name] = file_digest(path)
    return identity


def numerical_answer(text):
    # Strict final-answer extraction: reasoning numbers are never fallback answers.
    matches = re.findall(r"####\s*([-+]?(?:\d[\d,]*)(?:\.\d+)?)\s*(?:\n|$)", text)
    if not matches:
        return None
    try:
        return str(Decimal(matches[-1].replace(",", "")).normalize())
    except InvalidOperation:
        return None


def extract_code(text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    return blocks[0].strip() if blocks else text.strip()


# Installed inside an isolated interpreter before any generated code executes.
# Deny by default: no network, file writes, subprocesses, user/project file reads.
# This is a macOS-only local runner, not a portable security boundary for a service.
CHILD = r"""
import ctypes, json, os, resource, sys
payload = json.loads(sys.stdin.read())
resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2, 1024**2))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
output = sys.stdout
sys.stdout = sys.stderr = open(os.devnull, 'w')
lib = ctypes.CDLL('/usr/lib/libsandbox.dylib')
lib.sandbox_init.argtypes = [ctypes.c_char_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_char_p)]
error = ctypes.c_char_p()
policy = ('(version 1)(deny default)(allow sysctl-read)(allow file-read* (subpath '
          + json.dumps(sys.base_prefix)
          + ') (subpath "/System/Library") (subpath "/usr/lib"))'
          + '(allow file-write-data (literal ' + json.dumps(payload['output']) + '))')
if lib.sandbox_init(policy.encode(), 0, ctypes.byref(error)) != 0:
    output.write('SANDBOX_ERROR\n'); output.flush(); sys.exit(70)
try:
    exec(compile(payload['program'], '<candidate>', 'exec'), {'__name__': '__candidate__'})
except BaseException:
    output.write('FAIL\n')
else:
    output.write('PASS\n')
output.flush()
"""


def execute_code(program, timeout=8):
    if sys.platform != "darwin":
        raise ValueError("HumanEval execution requires macOS libsandbox; no unsafe fallback")
    import psutil

    # File-backed stdout plus RLIMIT_FSIZE bounds even direct os.write() output.
    # macOS does not reliably support RLIMIT_AS/DATA; monitor resident memory.
    with tempfile.TemporaryDirectory(prefix="actlogit-code-") as work:
        output_path = str(Path(work).resolve() / "result")
        with open(output_path, "w+b") as output, tempfile.TemporaryFile() as payload:
            payload.write(json.dumps({"program": program, "output": output_path}).encode())
            payload.seek(0)
            with subprocess.Popen(
                [sys.executable, "-I", "-S", "-c", CHILD],
                stdin=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                cwd=work,
                env={"LANG": "en_US.UTF-8"},
            ) as process:
                monitored = psutil.Process(process.pid)
                start = time.monotonic()
                while process.poll() is None:
                    reason = None
                    if time.monotonic() - start > timeout:
                        reason = "timeout"
                    try:
                        if monitored.memory_info().rss > 512 * 1024**2:
                            reason = "memory_limit"
                    except psutil.NoSuchProcess:
                        break
                    if reason:
                        process.kill()
                        process.wait()
                        return {"score": 0, "code_status": reason}
                    time.sleep(0.02)
                returncode = process.wait()
            output.seek(0)
            status = output.read(1024).decode(errors="replace").strip()
    if returncode == 70 or status == "SANDBOX_ERROR":
        raise ValueError("macOS code sandbox initialization failed")
    return {
        "score": int(returncode == 0 and status == "PASS"),
        "code_status": "pass" if returncode == 0 and status == "PASS" else "fail",
    }


def sandbox_preflight():
    program = """
import os, socket, subprocess
for action in [lambda: open(PROJECT_FILE).read(),
               lambda: open('/tmp/actlogit-sandbox-probe', 'w'),
               lambda: socket.socket().connect(('127.0.0.1', 9)),
               lambda: subprocess.run(['/usr/bin/true'])]:
    try:
        action()
    except PermissionError:
        pass
    else:
        raise AssertionError('sandbox probe was not denied')
"""
    program = "PROJECT_FILE = " + repr(str(Path(__file__).resolve())) + "\n" + program
    if execute_code(program)["score"] != 1:
        raise ValueError("HumanEval sandbox preflight failed; refusing to execute generated code")


def score(record, response):
    task = record["task"]
    if task in {"mmlu", "cmmlu"}:
        return {"score": int(response == "ABCD"[record["answer"]])}
    if task == "gsm8k":
        answer = numerical_answer(response)
        expected = numerical_answer(record["answer"])
        if expected is None:
            raise ValueError("GSM8K reference has no valid numerical answer")
        return {"score": int(answer is not None and answer == expected), "extracted_answer": answer}
    if task == "ifeval":
        from actlogit.regression._ifeval import evaluation_lib as official

        inp = official.InputExample(
            key=record["key"],
            instruction_id_list=record["instruction_id_list"],
            prompt=record["prompt"],
            kwargs=[{k: v for k, v in item.items() if v is not None} for item in record["kwargs"]],
        )
        random.seed(0)
        strict = official.test_instruction_following_strict(inp, {inp.prompt: response})
        random.seed(0)
        loose = official.test_instruction_following_loose(inp, {inp.prompt: response})
        return {
            "score": int(strict.follow_all_instructions),
            "prompt_loose": int(loose.follow_all_instructions),
            "instructions_strict": strict.follow_instruction_list,
            "instructions_loose": loose.follow_instruction_list,
        }
    if task == "humaneval":
        program = (
            extract_code(response) + "\n\n" + record["test"] + f"\ncheck({record['entry_point']})\n"
        )
        return execute_code(program)
    raise ValueError(f"unknown task: {task}")
