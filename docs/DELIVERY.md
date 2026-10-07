# Delivery ledger

- [x] Official archive/mirror availability verified
- [x] All files decoded, hashed and structurally audited
- [x] Whole-engine partitions and validation endpoints fixed
- [x] Classical/GRU training and shared inference implemented
- [x] Exact interrupted/resumed CPU training tested for both GRUs
- [x] Real-window CPU/MPS pilot performed
- [ ] Frozen 52-fit study completed
- [ ] Four-subset selections frozen before evaluation
- [ ] Official held-out clean/stress evaluation and uncertainty completed
- [ ] Real result figures and failure analysis generated
- [ ] Interactive UI, upload and exports verified
- [ ] Linux CI passed
- [ ] Checksum-pinned release assets published
- [ ] Actual hosted example/upload/download behavior verified
- [ ] v0.1.0 and measured CV/interview notes published

Quality claims and engineering completion are separate. The robustness hypothesis need not pass for the declared research release to be complete.

## Pre-publication findings

- Full local suite: 35 behavioral tests pass; real training-window smoke passed.
- The current study CLI predicts from its frozen study directory. Before publishing, provide a release-bundle path for clean-checkout prediction and a distinct inference benchmark command; preserve the original frozen source revision when adding these deployment conveniences.
- The GitHub repository exists, but the connected integration returned HTTP 403 when writing its first tree. The user has been asked to grant the new repository access. SSH authentication is unavailable. External publication is not yet claimed.
