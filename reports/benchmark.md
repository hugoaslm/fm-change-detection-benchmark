# LEVIR-CD representation benchmark

## Protocol

Feature-layer and anomaly-score selection used 256 training and 256 validation image pairs.
The selection process accessed only the `train` and `val` splits and evaluated 21 candidates.
Average precision was the primary selection criterion, followed by AUROC and calibrated F1.

One configuration was frozen for each representation family:

| Representation | Layer | Score | Validation AP | Validation AUROC |
|---|---|---|---:|---:|
| RGB pixels | input | cosine | 0.0453 | 0.5448 |
| ResNet-50, ImageNet | layer 3 | cosine | 0.2002 | 0.8987 |
| ResNet-50, SSL4EO-S12 MoCo | layer 3 | cosine | 0.1935 | 0.8743 |
| DINOv2 ViT-S/14 | block 3 | cosine | 0.2508 | 0.9168 |

The final run refitted Otsu and max-validation-F1 thresholds on the complete validation split.
These thresholds were then held fixed while evaluating all 2,048 test tiles. Encoders were
frozen throughout. The run used code revision `a5341fe`, seed 42, and dataset manifest
`2071cb9f6f84a9b1`.

## Test results

### Threshold-free separation

| Representation | AP | AUROC |
|---|---:|---:|
| RGB pixels | 0.0531 | 0.5281 |
| ResNet-50, ImageNet | 0.2074 | 0.8837 |
| ResNet-50, SSL4EO-S12 MoCo | 0.2043 | 0.8608 |
| **DINOv2 ViT-S/14** | **0.2406** | **0.8988** |

### Binary change maps

| Representation | Otsu F1 | Calibrated precision | Calibrated recall | Calibrated F1 | Calibrated IoU | Calibrated FPR |
|---|---:|---:|---:|---:|---:|---:|
| RGB pixels | 0.0351 | 0.0578 | 0.4324 | 0.1019 | 0.0537 | 0.3786 |
| ResNet-50, ImageNet | 0.1956 | 0.2119 | 0.6284 | 0.3169 | 0.1883 | 0.1255 |
| ResNet-50, SSL4EO-S12 MoCo | 0.1811 | 0.2093 | 0.5840 | 0.3081 | 0.1821 | 0.1184 |
| **DINOv2 ViT-S/14** | **0.2318** | **0.2396** | 0.6222 | **0.3460** | **0.2092** | **0.1060** |

## Findings

1. Frozen pretrained features provide substantially better change separation than direct RGB
   differencing. DINOv2 increased average precision from 0.0531 to 0.2406.
2. An early DINOv2 layer was preferable to the middle and final blocks considered during
   selection. This is consistent with the need to retain spatial detail while gaining
   invariance to incidental appearance differences.
3. DINOv2 gave the strongest final result: 0.2406 AP, 0.8988 AUROC, and 0.3460 calibrated F1.
   Relative to ImageNet ResNet-50, this corresponds to a 16.0% increase in AP and a 9.2%
   increase in calibrated F1.
4. SSL4EO-S12 MoCo pretraining did not outperform ImageNet pretraining for the matched
   ResNet-50 architecture. The difference is small enough that it should be interpreted as a
   result for these checkpoints and this dataset, not as a general statement about
   remote-sensing pretraining.
5. Otsu thresholds favored very high recall for the learned representations but produced many
   false positives. Validation-label calibration substantially improved F1 and IoU, showing
   that threshold choice remains an important part of anomaly-based change detection.

## Limitations

- The benchmark covers one building-change dataset and four representation families.
- LEVIR-CD contains registered, same-sensor RGB pairs and does not test heterogeneous imagery
  or severe spatial misalignment.
- Candidate selection used one deterministic 256-pair validation subset.
- The calibrated operating point requires validation masks; only the Otsu results are
  label-free.
- Reported metrics are pixel-level. Object-level correctness, boundary accuracy, and semantic
  transition labels were not evaluated.
- Confidence intervals and repeated selection runs were omitted under the available compute
  budget.
- The implemented T2-only robustness protocol has not yet been run on the frozen candidates.

The compact result record is stored in
[`results/final_summary.csv`](results/final_summary.csv), and the executable frozen
configuration is stored in [`../configs/final_selected.yaml`](../configs/final_selected.yaml).
