# Controlled-change detectability frontier

Synthetic additive changes of known magnitude and spatial extent are injected into regions of the T2 timestamp that the real labels mark as unchanged. Ground truth is exactly the injected region. Thresholds are fitted on the clean validation split only.

## `mock_encoder` / `layer1`

| Area | Magnitude | Samples | AP | AUROC | F1 (calib) | IoU (calib) | FPR (calib) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.010 | 0.050 | 8 | 0.0365 | 0.8650 | 0.0008 | 0.0004 | 0.1305 |
| 0.010 | 0.100 | 8 | 0.0367 | 0.8661 | 0.0098 | 0.0049 | 0.1309 |
| 0.010 | 0.200 | 8 | 0.0372 | 0.8682 | 0.0648 | 0.0335 | 0.1322 |
| 0.010 | 0.400 | 8 | 0.0386 | 0.8796 | 0.1331 | 0.0713 | 0.1330 |
| 0.040 | 0.050 | 8 | 0.1219 | 0.8590 | 0.0010 | 0.0005 | 0.1345 |
| 0.040 | 0.100 | 8 | 0.1225 | 0.8602 | 0.0234 | 0.0118 | 0.1357 |
| 0.040 | 0.200 | 8 | 0.1238 | 0.8621 | 0.1813 | 0.0997 | 0.1379 |
| 0.040 | 0.400 | 8 | 0.1286 | 0.8745 | 0.3670 | 0.2247 | 0.1392 |
