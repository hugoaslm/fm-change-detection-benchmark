# LEVIR-CD Zero-Shot Change Detection Benchmark Report

> Exploratory pipeline-validation run using 32 validation and 64 test image pairs. These
> results verify the implementation and must not be presented as the final benchmark.

## 1. Executive Summary
This report summarizes evaluation results across 8 experimental runs.

## 2. Benchmark Leaderboard

| Run ID | Encoder | Layer | Score | Threshold Method | AP (↑) | AUROC (↑) | F1 (↑) | IoU (↑) | FPR (↓) |
|---|---|---|---|---|---|---|---|---|---|
| `dinov2_vits14_block9_cosine_calibrated` | `dinov2_vits14` | `block9` | `cosine` | `calibrated` | 0.2457 | 0.8859 | 0.3702 | 0.2271 | 0.1237 |
| `dinov2_vits14_block9_cosine_unlabeled` | `dinov2_vits14` | `block9` | `cosine` | `unlabeled` | 0.2457 | 0.8859 | 0.2434 | 0.1386 | 0.3722 |
| `resnet50_imagenet_layer3_cosine_calibrated` | `resnet50_imagenet` | `layer3` | `cosine` | `calibrated` | 0.2355 | 0.8953 | 0.3642 | 0.2227 | 0.1672 |
| `resnet50_imagenet_layer3_cosine_unlabeled` | `resnet50_imagenet` | `layer3` | `cosine` | `unlabeled` | 0.2355 | 0.8953 | 0.2404 | 0.1366 | 0.3925 |
| `resnet50_s2_moco_layer3_cosine_calibrated` | `resnet50_s2_moco` | `layer3` | `cosine` | `calibrated` | 0.2317 | 0.8773 | 0.3476 | 0.2104 | 0.1640 |
| `resnet50_s2_moco_layer3_cosine_unlabeled` | `resnet50_s2_moco` | `layer3` | `cosine` | `unlabeled` | 0.2317 | 0.8773 | 0.2352 | 0.1333 | 0.3899 |
| `rgb_pixels_input_cosine_calibrated` | `rgb_pixels` | `input` | `cosine` | `calibrated` | 0.0604 | 0.5305 | 0.1109 | 0.0587 | 0.3347 |
| `rgb_pixels_input_cosine_unlabeled` | `rgb_pixels` | `input` | `cosine` | `unlabeled` | 0.0604 | 0.5305 | 0.0360 | 0.0183 | 0.0411 |

## 3. Scientific Subquestion Synthesis
1. **Remote-Sensing vs ImageNet**: Controlled ResNet-50 comparison.
2. **DINOv2 vs CNN**: Representation change separation capability.
3. **Nuisance Robustness**: Sensitivity to brightness, contrast, noise, blur, translation, saturation.
4. **Validation Calibration**: Improvement of validation max-F1 thresholding over unlabeled Otsu thresholding.
