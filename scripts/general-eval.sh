#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
python_bin="${ACTLOGIT_PYTHON:-$project_dir/.venv/bin/python}"
eval_root="${ACTLOGIT_EVAL_ROOT:-outputs/general-eval}"
adapter_dir="${ACTLOGIT_ADAPTER:-outputs/domain-adapter}"
config_path="${ACTLOGIT_CONFIG:-configs/mlx.toml}"
common=(--config "$config_path" --suite "$eval_root/suite.json"
        --nltk-data "$eval_root/nltk_data")

prepare() {
  "$python_bin" -m actlogit.cli general-eval prepare --output "$eval_root/suite.json" \
    --nltk-data "$eval_root/nltk_data"
}
base() {
  "$python_bin" -m actlogit.cli general-eval run "${common[@]}" --base \
    --output-dir "$eval_root/base"
}
trained() {
  "$python_bin" -m actlogit.cli general-eval run "${common[@]}" --adapter "$adapter_dir" \
    --output-dir "$eval_root/trained"
}
compare() {
  "$python_bin" -m actlogit.cli general-eval compare --base "$eval_root/base" \
    --adapter "$eval_root/trained" --output "$eval_root/comparison.json"
}
smoke() {
  prepare
  "$python_bin" -m actlogit.cli general-eval run "${common[@]}" --base \
    --output-dir "$eval_root/smoke-base" --limit-per-task 1 --max-new-tokens 64
  "$python_bin" -m actlogit.cli general-eval run "${common[@]}" --adapter "$adapter_dir" \
    --output-dir "$eval_root/smoke-adapter" --limit-per-task 1 --max-new-tokens 64
  "$python_bin" -m actlogit.cli general-eval compare --base "$eval_root/smoke-base" \
    --adapter "$eval_root/smoke-adapter" --output "$eval_root/smoke-comparison.json"
}

case "${1:-help}" in
  prepare) prepare ;;
  base) base ;;
  trained) trained ;;
  compare) compare ;;
  all) prepare; base; trained; compare ;;
  smoke) smoke ;;
  *) echo "Usage: bash scripts/general-eval.sh {prepare|base|trained|compare|all|smoke}"
     exit 2 ;;
esac
