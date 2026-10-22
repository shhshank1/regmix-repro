| Method | Test RMSE (nm), mean ± std over 5 seeds | Val RMSE (nm) | RegMix better by (nm) | paired t-test p |
|---|---|---|---|---|
| Vanilla | 13.908 ± 0.140 | 13.887 | +0.136 | 0.108 |
| Mixup (alpha=1, original default) | 13.986 ± 0.147 | 13.860 | +0.214 | 0.147 |
| Mixup (alpha=0.02, tuned) | 14.136 ± 0.137 | 13.908 | +0.363 | 0.013 |
| Manifold Mixup (alpha=0.02, tuned) | 14.004 ± 0.172 | 13.937 | +0.231 | 0.157 |
| Global kNN (k=8) | 13.876 ± 0.218 | 13.692 | +0.103 | 0.464 |
| Global Dist. (d=0.1) | 13.824 ± 0.126 | 13.622 | +0.052 | 0.616 |
| RegMix | 13.772 ± 0.111 | 13.508 | - | - |
