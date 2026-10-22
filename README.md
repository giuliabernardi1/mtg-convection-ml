# mtg-convection-ml

## Sources
https://gitlab.eumetsat.int/eumetlab/data-services/eumdac_data_store/-/tree/master

## Development
First attempt only with features on the single cell, unsatisfactory results.

=== fci_only ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.22   0.389      0.266            0.351
1  2026-07-19       0.26   0.320      0.239            0.258
2  2026-07-20       0.20   0.340      0.258            0.279
3  2026-07-21       0.20   0.298      0.245            0.232
4  2026-07-22       0.18   0.052      0.077            0.095

=== fci_plus_current_lightning ===
     test_day  threshold  pr_auc  model_csi  persistence_csi
0  2026-07-18       0.20   0.478      0.341            0.351
1  2026-07-19       0.22   0.378      0.278            0.258
2  2026-07-20       0.24   0.394      0.308            0.279
3  2026-07-21       0.24   0.339      0.273            0.232
4  2026-07-22       0.18   0.074      0.107            0.095

## Limitations
FCI data is every 30 minutes, while LI data is every 10 minutes. It was decided to accumulate the 3 LI data in the FCI time window.

A slight posiiton-wise shift is revealed between clouds and flashes. It is minimal but may be worth investigating.
