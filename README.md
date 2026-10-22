# mtg-convection-ml

## Sources
https://gitlab.eumetsat.int/eumetlab/data-services/eumdac_data_store/-/tree/master

## Limitations
FCI data is every 30 minutes, while LI data is every 10 minutes. It was decided to accumulate the 3 LI data in the FCI time window.

A slight posiiton-wise shift is revealed between clouds and flashes. It is minimal but may be worth investigating.
