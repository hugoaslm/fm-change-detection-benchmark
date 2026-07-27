"""Evaluation result recording and markdown report generation."""

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from fm_change_detection.metrics import MetricResults


def get_git_commit() -> str:
    """Retrieve short git commit hash or 'dirty'/'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        status_res = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=False
        )
        if status_res.stdout.strip():
            return f"{commit}-dirty"
        return commit
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def get_peak_gpu_memory() -> float:
    """Get peak GPU memory allocation in MB if CUDA is available."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return 0.0


def save_result_record(
    results_dir: str | Path,
    run_id: str,
    dataset: str,
    manifest_hash: str,
    encoder: str,
    checkpoint: str,
    layer: str,
    score_method: str,
    threshold_method: str,
    seed: int,
    metrics: MetricResults,
    runtime_seconds: float,
    additional_fields: dict[str, Any] | None = None,
) -> Path:
    """Save evaluation metrics atomically as JSON and append to CSV."""
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    peak_gpu_mb = get_peak_gpu_memory()

    record = {
        "run_id": run_id,
        "git_commit": git_commit,
        "dataset": dataset,
        "manifest_hash": manifest_hash,
        "encoder": encoder,
        "checkpoint": checkpoint,
        "layer": layer,
        "num_images": metrics.num_images,
        "score": score_method,
        "threshold_method": threshold_method,
        "threshold": float(metrics.threshold),
        "seed": seed,
        "num_pixels": metrics.num_pixels,
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "iou": float(metrics.iou),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "auroc": float(metrics.auroc),
        "average_precision": float(metrics.average_precision),
        "false_positive_rate": float(metrics.false_positive_rate),
        "runtime_seconds": float(runtime_seconds),
        "peak_gpu_memory_mb": float(peak_gpu_mb),
    }

    if metrics.ci_95:
        record["ci_95"] = {k: (float(v[0]), float(v[1])) for k, v in metrics.ci_95.items()}

    if additional_fields:
        record.update(additional_fields)

    # Save JSON atomically
    json_path = out_dir / f"{run_id}.json"
    tmp_json = out_dir / f"{run_id}.json.tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    tmp_json.replace(json_path)

    # Append to summary CSV atomically
    csv_path = out_dir / "summary.csv"
    file_exists = csv_path.exists()

    fieldnames = [
        "run_id",
        "git_commit",
        "dataset",
        "manifest_hash",
        "encoder",
        "checkpoint",
        "layer",
        "num_images",
        "score",
        "threshold_method",
        "threshold",
        "seed",
        "num_pixels",
        "precision",
        "recall",
        "f1",
        "iou",
        "balanced_accuracy",
        "auroc",
        "average_precision",
        "false_positive_rate",
        "runtime_seconds",
        "peak_gpu_memory_mb",
    ]

    csv_row = {k: record.get(k, "") for k in fieldnames}

    tmp_csv = out_dir / "summary.csv.tmp"
    if file_exists:
        # Read existing rows and append
        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        rows.append(csv_row)
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        tmp_csv.replace(csv_path)
    else:
        with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(csv_row)
        tmp_csv.replace(csv_path)

    return json_path


def generate_benchmark_report(results_dir: str | Path, output_file: str | Path) -> None:
    """Generate Markdown benchmark report from saved result JSON records."""
    res_dir = Path(results_dir)
    records = []

    if res_dir.exists():
        for json_file in sorted(res_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    records.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                pass

    out_file = Path(output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# LEVIR-CD Zero-Shot Change Detection Benchmark Report",
        "",
        "## 1. Executive Summary",
        f"This report summarizes evaluation results across {len(records)} experimental runs.",
        "",
        "## 2. Benchmark Leaderboard",
        "",
        "| Run ID | Encoder | Layer | Score | Threshold Method | AP (↑) | AUROC (↑) | F1 (↑) | IoU (↑) | FPR (↓) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    if not records:
        lines.append("| *No evaluation runs found* | | | | | | | | | |")
    else:
        for r in records:
            lines.append(
                f"| `{r.get('run_id', '')}` | `{r.get('encoder', '')}` | `{r.get('layer', '')}` | "
                f"`{r.get('score', '')}` | `{r.get('threshold_method', '')}` | "
                f"{r.get('average_precision', 0.0):.4f} | {r.get('auroc', 0.0):.4f} | "
                f"{r.get('f1', 0.0):.4f} | {r.get('iou', 0.0):.4f} | {r.get('false_positive_rate', 0.0):.4f} |"
            )

    lines.extend(
        [
            "",
            "## 3. Scientific Subquestion Synthesis",
            "1. **Remote-Sensing vs ImageNet**: Controlled ResNet-50 comparison.",
            "2. **DINOv2 vs CNN**: Representation change separation capability.",
            "3. **Nuisance Robustness**: Sensitivity to brightness, contrast, noise, blur, translation, saturation.",
            "4. **Validation Calibration**: Improvement of validation max-F1 thresholding over unlabeled Otsu thresholding.",
            "",
        ]
    )

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
