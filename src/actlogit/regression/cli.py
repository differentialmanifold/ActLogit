from __future__ import annotations


def register(commands):
    command = commands.add_parser("general-eval", help="600-question general-capability regression")
    actions = command.add_subparsers(dest="eval_action", required=True)
    prepare = actions.add_parser(
        "prepare", help="download pinned public data and prepare fixed suite"
    )
    prepare.add_argument("--output", default="outputs/general-eval/suite.json")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--nltk-data", default="outputs/general-eval/nltk_data")
    run = actions.add_parser("run", help="run or resume an MLX base/adapter evaluation")
    run.add_argument("--config", required=True)
    run.add_argument("--suite", default="outputs/general-eval/suite.json")
    run.add_argument("--nltk-data", default="outputs/general-eval/nltk_data")
    arm = run.add_mutually_exclusive_group(required=True)
    arm.add_argument(
        "--base", action="store_true", help="explicitly disable any configured adapter"
    )
    arm.add_argument("--adapter", help="ActLogit MLX adapter directory")
    run.add_argument("--output-dir", required=True)
    run.add_argument(
        "--limit-per-task", type=int, help="pilot only: use first N selected tasks each"
    )
    run.add_argument(
        "--max-new-tokens", type=int, help="smoke only: lower all generation token caps"
    )
    compare = actions.add_parser(
        "compare", help="compare two completed, identical evaluation protocols"
    )
    compare.add_argument("--base", required=True)
    compare.add_argument("--adapter", required=True)
    compare.add_argument("--output", default="outputs/general-eval/comparison.json")


def dispatch(args):
    if args.eval_action == "prepare":
        from actlogit.regression.scoring import prepare_checker, sandbox_preflight
        from actlogit.regression.suite import prepare

        sandbox_preflight()
        prepare_checker(args.nltk_data)
        return prepare(args.output, args.seed)
    if args.eval_action == "run":
        from actlogit.regression.runner import run

        return run(args)
    from actlogit.regression.report import compare

    return compare(args.base, args.adapter, args.output)
