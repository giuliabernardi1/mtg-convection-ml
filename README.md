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

Adding days to try and improve performance:
- very active days -> 1st, 11th, 15th and 26th of July
- average active days -> 3rd, 24th and 31st of July
- barely active days -> 9th, 13th, and 30th of July

=== fci_only ===
   test_block                        test_days  threshold  pr_auc  model_csi  persistence_csi
0           0  2026-07-01..2026-07-03 (2 days)       0.20   0.384      0.273            0.427
1           1  2026-07-09..2026-07-15 (4 days)       0.28   0.297      0.232            0.335
2           2  2026-07-18..2026-07-26 (7 days)       0.18   0.287      0.222            0.275
3           3  2026-07-30..2026-07-31 (2 days)       0.22   0.369      0.269            0.351
model {'pod': np.float64(0.497), 'far': np.float64(0.691), 'csi': np.float64(0.236)}
persistence {'pod': np.float64(0.492), 'far': np.float64(0.505), 'csi': np.float64(0.328)}

=== fci_plus_current_lightning ===
   test_block                        test_days  threshold  pr_auc  model_csi  persistence_csi
0           0  2026-07-01..2026-07-03 (2 days)       0.24   0.536      0.415            0.427
1           1  2026-07-09..2026-07-15 (4 days)       0.24   0.437      0.326            0.335
2           2  2026-07-18..2026-07-26 (7 days)       0.26   0.365      0.295            0.275
3           3  2026-07-30..2026-07-31 (2 days)       0.24   0.493      0.363            0.351
model {'pod': np.float64(0.542), 'far': np.float64(0.536), 'csi': np.float64(0.333)}
persistence {'pod': np.float64(0.492), 'far': np.float64(0.505), 'csi': np.float64(0.328)}

=== fci_plus_neighbourhood ===
   test_block                        test_days  threshold  pr_auc  model_csi  persistence_csi
0           0  2026-07-01..2026-07-03 (2 days)       0.24   0.462      0.325            0.427
1           1  2026-07-09..2026-07-15 (4 days)       0.22   0.353      0.264            0.335
2           2  2026-07-18..2026-07-26 (7 days)       0.22   0.356      0.266            0.275
3           3  2026-07-30..2026-07-31 (2 days)       0.24   0.431      0.293            0.351
model {'pod': np.float64(0.557), 'far': np.float64(0.641), 'csi': np.float64(0.279)}
persistence {'pod': np.float64(0.492), 'far': np.float64(0.505), 'csi': np.float64(0.328)}

=== fci_plus_lightning_intensity ===
   test_block                        test_days  threshold  pr_auc  model_csi  persistence_csi
0           0  2026-07-01..2026-07-03 (2 days)       0.24   0.577      0.419            0.427
1           1  2026-07-09..2026-07-15 (4 days)       0.24   0.484      0.350            0.335
2           2  2026-07-18..2026-07-26 (7 days)       0.26   0.397      0.310            0.275
3           3  2026-07-30..2026-07-31 (2 days)       0.26   0.530      0.389            0.351
model {'pod': np.float64(0.579), 'far': np.float64(0.534), 'csi': np.float64(0.348)}
persistence {'pod': np.float64(0.492), 'far': np.float64(0.505), 'csi': np.float64(0.328)}

=== all_features ===
   test_block                        test_days  threshold  pr_auc  model_csi  persistence_csi
0           0  2026-07-01..2026-07-03 (2 days)       0.24   0.621      0.429            0.427
1           1  2026-07-09..2026-07-15 (4 days)       0.30   0.520      0.374            0.335
2           2  2026-07-18..2026-07-26 (7 days)       0.30   0.428      0.317            0.275
3           3  2026-07-30..2026-07-31 (2 days)       0.26   0.555      0.391            0.351
model {'pod': np.float64(0.594), 'far': np.float64(0.52), 'csi': np.float64(0.361)}
persistence {'pod': np.float64(0.492), 'far': np.float64(0.505), 'csi': np.float64(0.328)}

The ML model did not improve much.

## Limitations
FCI data is every 30 minutes, while LI data is every 10 minutes. It was decided to accumulate the 3 LI data in the FCI time window.

A slight posiiton-wise shift is revealed between clouds and flashes. It is minimal but may be worth investigating.
