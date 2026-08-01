# Benchmarking Frozen Representations for Change Detection

A reproducible benchmark of frozen visual representations for anomaly-based building change
detection in bitemporal satellite imagery.

Given two registered images, the benchmark extracts spatial feature maps with a frozen encoder
and measures their pixel-aligned cosine distance. A validation-derived threshold converts the
resulting anomaly map into a binary change map. The study compares raw RGB values, ImageNet
features, remote-sensing self-supervised features, and DINOv2 features on LEVIR-CD.

## Results

The representation, feature layer, and distance function were selected without accessing the
test split. The frozen configurations were then evaluated on all 2,048 test tiles.

| Representation | Selected layer | Test AP | Test AUROC | Calibrated F1 | Calibrated IoU |
|---|---:|---:|---:|---:|---:|
| RGB pixels | input | 0.0531 | 0.5281 | 0.1019 | 0.0537 |
| ResNet-50, ImageNet | layer 3 | 0.2074 | 0.8837 | 0.3169 | 0.1883 |
| ResNet-50, SSL4EO-S12 MoCo | layer 3 | 0.2043 | 0.8608 | 0.3081 | 0.1821 |
| **DINOv2 ViT-S/14** | **block 3** | **0.2406** | **0.8988** | **0.3460** | **0.2092** |

AP and AUROC are threshold-free. F1 and IoU use a max-F1 threshold fitted on the complete
validation split and held fixed during test evaluation.

DINOv2 block 3 produced the strongest separation: its test AP was 0.2406, compared with 0.2074
for the ImageNet ResNet-50 and 0.2043 for the remote-sensing ResNet-50. All learned
representations substantially outperformed direct RGB differencing. In this experiment,
remote-sensing-specific pretraining did not improve on ImageNet pretraining under a controlled
ResNet-50 comparison.

The complete selection table and test records are documented in
[`reports/benchmark.md`](reports/benchmark.md). The compact machine-readable results
are available in [`reports/results/final_summary.csv`](reports/results/final_summary.csv).

## Experimental protocol

### Dataset

LEVIR-CD contains registered RGB image pairs with binary labels for building appearance and
disappearance. Images are evaluated as 256 × 256 tiles; encoder inputs are resized to
252 × 252 and score maps are interpolated back to label resolution.

Download the dataset for academic, noncommercial use from the
[official LEVIR-CD page](https://justchenhao.github.io/LEVIR/). The expected layout is:

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

Datasets, checkpoints, feature caches, and raw experiment outputs are excluded from version
control.

### Representation selection

Selection used 256 training and 256 validation pairs and did not load the test split. The
candidate matrix comprised:

- raw RGB values;
- ResNet-50 pretrained on ImageNet;
- ResNet-50 pretrained with MoCo on SSL4EO-S12 Sentinel-2 RGB;
- DINOv2 ViT-S/14;
- early, middle, and late feature layers where available;
- cosine and channel-standardized Euclidean distances.

Within each representation family, the layer and score were selected by validation average
precision, with AUROC and calibrated F1 as deterministic tie-breakers. This reduced 21
candidates to one frozen configuration per representation.

### Final evaluation

The final benchmark refitted thresholds on the complete validation split, then evaluated the
four frozen configurations on all 2,048 test tiles. Two thresholding regimes are reported:

- **unlabeled:** Otsu threshold estimated from validation anomaly scores without labels;
- **calibrated:** max-F1 threshold estimated from validation scores and masks.

No model, layer, distance, or threshold was selected using test labels. All learned encoders
remain in evaluation mode with gradients disabled.

### Detectability frontier (experimental)

A controlled-change experiment that maps *where each frozen representation stops working*.
Synthetic additive changes of known intensity and spatial extent are injected into regions of
the T2 timestamp that the real labels mark as unchanged, producing exactly controlled ground
truth. Detection metrics are measured with the clean validation-fitted threshold held fixed,
so performance drops reflect the representation's separation capacity rather than threshold
recalibration.

For each encoder and for every combination of change magnitude and change area, the benchmark
reports threshold-free AP/AUROC plus F1/IoU at the frozen clean threshold. The result is an
operating-characteristic-style table (`reports/frontier.md`), a machine-readable CSV, and an
AP-vs-magnitude curve per area fraction for each encoder (`reports/figures/frontier_*.png`).

```bash
uv run fmcd frontier --config configs/detectability.yaml --data-root /path/to/LEVIR-CD
```

## Reproduction

Python 3.11 and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
uv sync --extra dev --frozen
uv run fmcd smoke
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

The smoke test uses synthetic data, downloads no checkpoints, and runs on CPU.

Validate a local LEVIR-CD installation:

```bash
uv run fmcd validate-data --root /path/to/LEVIR-CD
```

Run validation-only representation selection:

```bash
uv run fmcd select \
  --config configs/selection.yaml \
  --data-root /path/to/LEVIR-CD \
  --device cuda
```

After reviewing and freezing the selection, run the complete benchmark:

```bash
uv run fmcd benchmark \
  --config configs/final_selected.yaml \
  --data-root /path/to/LEVIR-CD \
  --device cuda
```

The Colab workflow in
[`notebooks/colab_quickstart.ipynb`](notebooks/colab_quickstart.ipynb) covers dataset
validation, resumable selection, selection verification, and the frozen test run.

## Implementation

- common frozen-encoder interface with explicit layer extraction;
- deterministic feature caches keyed by dataset manifest, checkpoint, preprocessing, and layer;
- token-to-grid conversion and spatial score-map alignment;
- cosine and channel-standardized Euclidean anomaly scores;
- validation-only Otsu and max-F1 threshold fitting;
- pixel-level precision, recall, F1, IoU, balanced accuracy, AUROC, and average precision;
- T2-only nuisance perturbations with frozen clean thresholds;
- controlled synthetic changes (region sampling + additive offsets) with frozen clean
  thresholds for detectability-frontier analysis;
- CPU unit and synthetic end-to-end tests.

## Limitations

- LEVIR-CD measures building changes only; the findings do not establish performance on other
  land-cover transitions, sensors, or geographic domains.
- Images are registered RGB pairs. Heterogeneous sensors and substantial misregistration are
  outside the present evaluation.
- The layer and score search used a deterministic 256-pair validation subset. Repeating
  selection across alternative subsets would better characterize selection variance.
- The calibrated threshold uses validation labels and is therefore a low-label calibration
  setting rather than fully unsupervised detection. Otsu results provide the unlabeled
  counterpart.
- Metrics are pixel-level and do not directly measure object detection, boundary quality, or
  the semantic nature of a transition.
- Confidence intervals, multi-seed analysis, and the planned nuisance-robustness study are not
  included in the reported v0.1 results.
- Only one remote-sensing-specific checkpoint was evaluated; its result should not be treated
  as a general comparison between domain-specific and generic pretraining.

## License

The code is released under the MIT License. Dataset and pretrained-checkpoint licenses remain
separate and are not superseded by the repository license.
