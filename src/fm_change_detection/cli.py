"""Command-Line Interface (CLI) for fm_change_detection."""

import argparse
import sys

from fm_change_detection.config import load_config
from fm_change_detection.data import validate_dataset_layout
from fm_change_detection.pipeline import (
    run_benchmark,
    run_robustness,
    run_single_evaluation,
    run_smoke_test,
)
from fm_change_detection.reporting import generate_benchmark_report
from fm_change_detection.selection import run_validation_selection


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", help="Override dataset.root from the YAML config")
    parser.add_argument("--device", help="Execution device: auto, cpu, cuda, or cuda:0")
    parser.add_argument("--max-train-samples", type=int, help="Deterministic training subset size")
    parser.add_argument("--max-val-samples", type=int, help="Deterministic validation subset size")
    parser.add_argument("--max-test-samples", type=int, help="Deterministic test subset size")


def _apply_runtime_overrides(cfg, args) -> None:
    if getattr(args, "data_root", None):
        cfg.dataset.root = args.data_root
    if getattr(args, "device", None):
        cfg.runtime.device = args.device
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg.runtime, name, value)


def main() -> None:
    """CLI entrypoint for fmcd tool."""
    parser = argparse.ArgumentParser(
        prog="fmcd",
        description="Foundation Model Change Detection Benchmark CLI (v0.1)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. smoke
    parser_smoke = subparsers.add_parser("smoke", help="Run CPU synthetic end-to-end smoke test")
    parser_smoke.add_argument(
        "--config", default="configs/smoke.yaml", help="Path to smoke config YAML"
    )

    # 2. validate-data
    parser_valdata = subparsers.add_parser(
        "validate-data", help="Validate LEVIR-CD directory layout and split lists"
    )
    parser_valdata.add_argument(
        "--root", default="data/raw/LEVIR-CD", help="Path to LEVIR-CD root directory"
    )

    # 3. extract
    parser_extract = subparsers.add_parser(
        "extract", help="Pre-extract and cache features for an encoder"
    )
    parser_extract.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to config YAML"
    )
    parser_extract.add_argument(
        "--encoder", required=True, help="Encoder name (e.g. dinov2_vits14)"
    )
    _add_runtime_arguments(parser_extract)

    # 4. evaluate
    parser_eval = subparsers.add_parser(
        "evaluate", help="Evaluate single encoder, layer, and score method"
    )
    parser_eval.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to config YAML"
    )
    parser_eval.add_argument("--encoder", required=True, help="Encoder name (e.g. dinov2_vits14)")
    parser_eval.add_argument("--layer", required=True, help="Layer name (e.g. block9)")
    parser_eval.add_argument(
        "--score", default="cosine", help="Score method: cosine or standardized_euclidean"
    )
    _add_runtime_arguments(parser_eval)

    # 5. benchmark
    parser_select = subparsers.add_parser(
        "select",
        help="Select one layer and score per encoder using train/validation data only",
    )
    parser_select.add_argument(
        "--config", default="configs/selection.yaml", help="Path to selection config YAML"
    )
    _add_runtime_arguments(parser_select)

    # 6. benchmark
    parser_bench = subparsers.add_parser(
        "benchmark", help="Run full benchmark across all configured encoders/layers"
    )
    parser_bench.add_argument(
        "--config", default="configs/baseline.yaml", help="Path to config YAML"
    )
    _add_runtime_arguments(parser_bench)

    # 7. robustness
    parser_rob = subparsers.add_parser("robustness", help="Run robustness perturbation experiments")
    parser_rob.add_argument(
        "--config", default="configs/robustness.yaml", help="Path to robustness config YAML"
    )
    _add_runtime_arguments(parser_rob)

    # 8. report
    parser_report = subparsers.add_parser("report", help="Generate Markdown benchmark report")
    parser_report.add_argument(
        "--results", default="outputs/results", help="Directory containing result JSON files"
    )
    parser_report.add_argument(
        "--output", default="reports/benchmark.md", help="Output markdown report file"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "smoke":
        cfg = load_config(args.config)
        run_smoke_test(cfg)

    elif args.command == "validate-data":
        counts = validate_dataset_layout(args.root)
        print(f"[DATA] LEVIR-CD dataset layout valid at '{args.root}':")
        for split, count in counts.items():
            print(f"  - {split}: {count} samples")

    elif args.command == "extract":
        cfg = load_config(args.config)
        _apply_runtime_overrides(cfg, args)
        print(f"[EXTRACT] Extracting features for encoder '{args.encoder}'...")
        # A cached evaluation warms validation/test features for the selected layer.
        enc_cfg = next((e for e in cfg.encoders if e.name == args.encoder), None)
        layer = enc_cfg.layers[0] if enc_cfg and enc_cfg.layers else "layer4"
        run_single_evaluation(cfg, args.encoder, layer_name=layer, score_method="cosine")

    elif args.command == "evaluate":
        cfg = load_config(args.config)
        _apply_runtime_overrides(cfg, args)
        run_single_evaluation(cfg, args.encoder, layer_name=args.layer, score_method=args.score)

    elif args.command == "select":
        cfg = load_config(args.config)
        _apply_runtime_overrides(cfg, args)
        result = run_validation_selection(cfg)
        print(f"[SELECT] Validation-only report: {result['report_path']}")
        print(f"[SELECT] Frozen final config: {result['final_config_path']}")

    elif args.command == "benchmark":
        cfg = load_config(args.config)
        _apply_runtime_overrides(cfg, args)
        run_benchmark(cfg)

    elif args.command == "robustness":
        cfg = load_config(args.config)
        _apply_runtime_overrides(cfg, args)
        run_robustness(cfg)

    elif args.command == "report":
        generate_benchmark_report(args.results, args.output)
        print(f"[REPORT] Benchmark report updated at '{args.output}'")


if __name__ == "__main__":
    main()
