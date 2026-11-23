"""Command-line entrypoint; no network, GPU or model needed by default."""

import argparse
import importlib
import json
from pathlib import Path

from .data import read_csv
from .evaluation import evaluate, load_predictions, hallucination_rate
from .pipeline import run, validate_submission
from .schema import Schema
from .synthetic import make_synthetic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    synthetic = commands.add_parser("make-synthetic")
    synthetic.add_argument("--out", required=True)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--count", type=int, default=60)
    runner = commands.add_parser("run")
    runner.add_argument("--data-dir", required=True)
    runner.add_argument("--out", required=True)
    runner.add_argument("--input", default="Test_Masked.csv")
    runner.add_argument("--config", help="JSON with source_columns and field_aliases")
    runner.add_argument("--ocr-plugin", help="Trusted installed module:function taking an image path")
    runner.add_argument("--llm-endpoint", help="Local OpenAI-compatible chat completions URL only")
    runner.add_argument("--model", default="local-open-model")
    runner.add_argument("--profile", choices=("auto", "dischem", "generic"), default="auto",
                        help="auto selects the Dis-Chem column adapter when its input columns are present")
    evaluator = commands.add_parser("evaluate")
    evaluator.add_argument("--pred", required=True)
    evaluator.add_argument("--ref", required=True)
    evaluator.add_argument("--schema", help="Defaults to submission_format.json next to reference")
    evaluator.add_argument("--sentence-model", help="Optional local downloaded open model path")
    validator = commands.add_parser("validate-submission")
    validator.add_argument("--pred", required=True)
    validator.add_argument("--data-dir", required=True)
    args = parser.parse_args()
    if args.command == "make-synthetic":
        if args.count < 1:
            parser.error("--count must be positive")
        print(make_synthetic(args.out, args.seed, args.count))
    elif args.command == "run":
        config = json.loads(Path(args.config).read_text()) if args.config else None
        ocr = extractor = None
        if args.ocr_plugin:
            module, function = args.ocr_plugin.split(":")
            ocr = getattr(importlib.import_module(module), function)
        if args.llm_endpoint:
            from .plugins import LocalLLM
            extractor = LocalLLM(args.llm_endpoint, args.model)
        rows = run(args.data_dir, args.out, args.input, config, ocr, extractor, args.profile)
        print(f"Wrote {len(rows)} products to {args.out}")
    elif args.command == "evaluate":
        schema = Schema.load(args.schema or Path(args.ref).parent / "submission_format.json")
        predictions, references = load_predictions(args.pred, schema), load_predictions(args.ref, schema)
        kwargs = {}
        if args.sentence_model:
            from .plugins import sentence_similarity
            kwargs["similarity"] = sentence_similarity(args.sentence_model)
        result = evaluate(predictions, references, schema, **kwargs)
        result["hallucination"] = hallucination_rate(predictions, references, schema)
        print(json.dumps(result, indent=2))
    else:
        root = Path(args.data_dir)
        errors = validate_submission(args.pred, Schema.load(root / "submission_format.json"),
                                     root / "SampleSubmission.csv", read_csv(root / "Test_Masked.csv"))
        if errors:
            parser.exit(1, "\n".join(errors) + "\n")
        print("Submission valid")


if __name__ == "__main__":
    main()
