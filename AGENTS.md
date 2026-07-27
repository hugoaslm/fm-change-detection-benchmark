# AGENTS.md

## Overview
This repository implements a rigorous, scientifically defensible benchmark (v0.1) comparing feature representations for zero-shot satellite image change detection on LEVIR-CD.

## Guidelines for AI Agents
1. **Inference Mode**: All encoders must run in evaluation mode (`eval()`, `requires_grad_(False)`).
2. **Threshold Secrecy**: Test thresholds must be fitted ONLY on validation data (Otsu or max-validation-F1). Never fit thresholds on test data.
3. **Robustness Protocol**: Perturbations (brightness, contrast, noise, blur, translation, saturation) apply to $T_2$ only, and re-use the clean validation threshold without recalibrating.
4. **Resumable Caching**: Feature extractions are cached under `outputs/cache/` using deterministic configuration hashes.
5. **Quality Gate**: All changes must pass `ruff check .`, `ruff format --check .`, and `pytest`.
