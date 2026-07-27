# LEVIR-CD Dataset Protocol

Download LEVIR-CD from the [official project page](https://justchenhao.github.io/LEVIR/)
and use the train/validation/test lists distributed with the official cropped dataset. The
dataset is not redistributed by this repository.

## Expected Directory Layout
Place the official LEVIR-CD dataset under `data/raw/LEVIR-CD/` with the following structure:

```
data/raw/LEVIR-CD/
├── A/             # Image timestamp T1 (.png)
├── B/             # Image timestamp T2 (.png)
├── label/         # Binary change labels (.png)
└── list/
    ├── train.txt  # List of training image filenames
    ├── val.txt    # List of validation image filenames
    └── test.txt   # List of test image filenames
```

## Scene-Safe Tiling Protocol
1. Assign scenes (original 1024x1024) to train/val/test splits *before* tiling.
2. Generate non-overlapping 256x256 tiles per scene.
3. Retain empty change tiles (essential for true negative / FPR evaluation).
4. Both image timestamps are resized to common input dimension $252 \times 252$ for encoder forward pass, and extracted score maps are interpolated back to $256 \times 256$ native label resolution.

Run the strict validator before extracting any model features:

```bash
uv run fmcd validate-data --root /path/to/LEVIR-CD
```

The validator checks all referenced A/B/label files, rejects empty or duplicate split lists,
and rejects filenames that occur in more than one split.

## Google Colab

Keep the dataset in Google Drive rather than copying it into the Git repository. The Colab
notebook accepts the Drive directory through `--data-root`. Expected storage is several GB
including cached float16 features and downloaded model checkpoints.
