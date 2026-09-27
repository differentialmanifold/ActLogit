from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from actlogit import __version__
from actlogit.config import load_config
from actlogit.data import load_records
from actlogit.schema import SystemOneRequest
from actlogit.systemone import predict


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="actlogit", description="Local single-token decisions and forward-KL LoRA training."
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    from actlogit.regression.cli import register

    register(commands)
    validate = commands.add_parser("validate-data", help="validate JSONL without loading a model")
    validate.add_argument("--data", required=True)
    for name in ("predict", "train", "evaluate", "serve"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument(
            "--adapter", help="load an ActLogit adapter (overrides model.adapter_path)"
        )
        if name == "predict":
            command.add_argument(
                "--input", required=True, help="System One request JSON path, or - for stdin"
            )
        if name == "evaluate":
            command.add_argument("--data", required=True)
        if name == "serve":
            command.add_argument("--host", default="127.0.0.1")
            command.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    try:
        if args.command == "general-eval":
            from actlogit.regression.cli import dispatch

            result = dispatch(args)
        elif args.command == "validate-data":
            records = load_records(args.data)
            result = {
                "records": len(records),
                "question_types": sorted({r.question.type for r in records}),
                "valid": True,
            }
        else:
            config = load_config(args.config)
            if args.adapter:
                config.model.adapter_path = args.adapter
            if args.command == "train":
                from actlogit.backend import train

                result = train(config)
            else:
                from actlogit.backend import load_engine

                engine = load_engine(config)
                if args.command == "predict":
                    raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
                    result = predict(engine, SystemOneRequest.model_validate_json(raw)).model_dump()
                elif args.command == "evaluate":
                    from actlogit.backend import evaluate_records

                    result = evaluate_records(engine, load_records(args.data))
                else:
                    import uvicorn

                    from actlogit.server import create_app

                    uvicorn.run(create_app(engine), host=args.host, port=args.port)
                    return
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"actlogit: {exc}\n")
    except ImportError as exc:
        parser.exit(
            2,
            f"actlogit: missing dependency ({exc}); install the relevant "
            "mlx/eval/local/server extras\n",
        )


if __name__ == "__main__":
    main()
