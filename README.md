# mtg-convection-ml

## Sources
https://gitlab.eumetsat.int/eumetlab/data-services/eumdac_data_store/-/tree/master

## Development
First attempt only with features on the single cell.

=== fci_only ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.22   0.389      0.266            0.351
1  2026-07-19       0.26   0.320      0.239            0.258
2  2026-07-20       0.20   0.340      0.258            0.279
3  2026-07-21       0.20   0.298      0.245            0.232
4  2026-07-22       0.18   0.052      0.077            0.095
model {'pod': np.float64(0.47), 'far': np.float64(0.656), 'csi': np.float64(0.248)}
persistence {'pod': np.float64(0.423), 'far': np.float64(0.575), 'csi': np.float64(0.269)}

=== fci_plus_current_lightning ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.20   0.478      0.341            0.351
1  2026-07-19       0.22   0.378      0.278            0.258
2  2026-07-20       0.24   0.394      0.308            0.279
3  2026-07-21       0.24   0.339      0.273            0.232
4  2026-07-22       0.18   0.074      0.107            0.095
model {'pod': np.float64(0.527), 'far': np.float64(0.605), 'csi': np.float64(0.292)}
persistence {'pod': np.float64(0.423), 'far': np.float64(0.575), 'csi': np.float64(0.269)}

Second attempt adding also the neighbouring cells and the current and previous flash cumulate.

=== fci_plus_neighbourhood ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.22   0.442      0.305            0.351
1  2026-07-19       0.24   0.380      0.288            0.258
2  2026-07-20       0.22   0.410      0.303            0.279
3  2026-07-21       0.24   0.343      0.278            0.232
4  2026-07-22       0.22   0.089      0.108            0.095
model {'pod': np.float64(0.55), 'far': np.float64(0.625), 'csi': np.float64(0.287)}
persistence {'pod': np.float64(0.423), 'far': np.float64(0.575), 'csi': np.float64(0.269)}

=== fci_plus_lightning_intensity ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.20   0.500      0.366            0.351
1  2026-07-19       0.20   0.397      0.294            0.258
2  2026-07-20       0.24   0.443      0.334            0.279
3  2026-07-21       0.24   0.357      0.276            0.232
4  2026-07-22       0.22   0.094      0.108            0.095
model {'pod': np.float64(0.551), 'far': np.float64(0.593), 'csi': np.float64(0.306)}
persistence {'pod': np.float64(0.423), 'far': np.float64(0.575), 'csi': np.float64(0.269)}

Attempt with all the above features.

=== all_features ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.26   0.498      0.364            0.351
1  2026-07-19       0.24   0.445      0.333            0.258
2  2026-07-20       0.22   0.473      0.366            0.279
3  2026-07-21       0.22   0.384      0.303            0.232
4  2026-07-22       0.26   0.106      0.120            0.095
model {'pod': np.float64(0.609), 'far': np.float64(0.585), 'csi': np.float64(0.328)}
persistence {'pod': np.float64(0.423), 'far': np.float64(0.575), 'csi': np.float64(0.269)}

This is the best result up until this point.

## Limitations
FCI data is every 30 minutes, while LI data is every 10 minutes. It was decided to accumulate the 3 LI data in the FCI time window.

A slight posiiton-wise shift is revealed between clouds and flashes. It is minimal but may be worth investigating.
