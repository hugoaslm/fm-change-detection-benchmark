# Benchmarking Frozen Representations for Change Detection

A reproducible, low-compute benchmark of frozen RGB, ImageNet, remote-sensing, and DINOv2
representations for anomaly-based building change detection on LEVIR-CD.

## Research question

Which pretrained spatial representation best separates real building change from unchanged
regions while remaining robust to nuisance variation?

The benchmark currently compares:

- raw RGB pixels;
- ResNet-50 pretrained on ImageNet;
- ResNet-50 pretrained with MoCo on SSL4EO-S12 Sentinel-2 RGB;
- DINOv2 ViT-S/14.

LEVIR-CD labels building appearance and disappearance. Results must not be generalized to
arbitrary semantic land-cover transitions.

## Current status

The synthetic CPU smoke test and a bounded Colab run have passed. The initial 32-validation /
64-test exploratory run showed that all three learned representations substantially outperform
raw RGB. These numbers validate the pipeline but are not the final benchmark.

## Local installation and smoke test

Python 3.11 and `uv` are recommended.

```bash
uv sync --extra dev --frozen
uv run fmcd smoke
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The smoke test downloads neither LEVIR-CD nor pretrained checkpoints and is safe to run on CPU.

## Dataset

Download LEVIR-CD for academic, noncommercial use from the
[official dataset page](https://justchenhao.github.io/LEVIR/). The benchmark expects the
standard processed 256×256 layout:

```text
LEVIR-CD/
├── A/
├── B/
├── label/
└── list/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Validate it before any experiment:

```bash
uv run fmcd validate-data --root /path/to/LEVIR-CD
```

Datasets, checkpoints, feature caches, and raw outputs are intentionally excluded from Git.

## Experiment sequence

### 1. Bounded pipeline validation

```bash
uv run fmcd benchmark \
  --config configs/colab_quickstart.yaml \
  --data-root /path/to/LEVIR-CD \
  --device cuda
```

### 2. Validation-only layer and score selection

```bash
uv run fmcd select \
  --config configs/selection.yaml \
  --data-root /path/to/LEVIR-CD \
  --device cuda
```

The default selection run uses 256 train and 256 validation pairs. It compares the configured
layers with cosine and standardized Euclidean anomaly scores, selects by validation average
precision, and uses AUROC then calibrated F1 as deterministic tie-breakers.

The selection command never opens the test split and writes:

```text
outputs/selection/
├── candidates.csv
├── selection.json
├── selection.md
└── final_selected.yaml
```

### 3. Frozen full evaluation

Review and commit `final_selected.yaml` before evaluating the complete validation and test
splits. Bootstrap is deliberately disabled in the generated configuration until the scalable
full-test bootstrap stage is implemented.

### 4. Robustness

After the clean final benchmark, apply brightness, contrast, noise, blur, translation, and
saturation perturbations to T2 only. Every perturbed evaluation must reuse its clean validation
threshold without recalibration.

## Scientific safeguards

- Every learned encoder runs with `eval()` and `requires_grad_(False)`.
- Otsu and max-F1 thresholds are fitted on validation scores only.
- Model layer and anomaly-score selection uses train/validation data only.
- The test split is reserved for the frozen final configuration.
- Robustness perturbations affect T2 only and reuse clean thresholds.
- Feature caches use deterministic hashes containing the dataset manifest, checkpoint, layers,
  input size, normalization, and cache dtype.
- The remote-sensing model uses the pinned TorchGeo SSL4EO-S12 checkpoint rather than relabeled
  ImageNet weights.

## Quality gates

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## License and attribution

Repository code is MIT licensed. Dataset and pretrained-checkpoint licenses remain separate;
the repository license does not override them.
