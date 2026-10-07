# Delivery ledger

- [x] Official archive/mirror availability verified
- [x] All files decoded, hashed and structurally audited
- [x] Whole-engine partitions and validation endpoints fixed
- [x] Classical/GRU training and shared inference implemented
- [x] Exact interrupted/resumed CPU training tested for both GRUs
- [x] Real-window CPU/MPS pilot performed
- [x] Frozen 52-fit study completed
- [x] Four-subset selections frozen before evaluation
- [x] Official held-out clean/stress evaluation and uncertainty completed
- [x] Real result figures and failure analysis generated
- [ ] Interactive UI, upload and exports verified
- [x] Linux CI passed
- [x] Checksum-pinned release assets published
- [ ] Actual hosted example/upload/download behavior verified
- [x] v0.1.0 and measured CV/interview notes published

Quality claims and engineering completion are separate. The robustness hypothesis need not pass for the declared research release to be complete.

## Verified publication

All 52 fits, official evaluation and 2,000-engine-bootstrap reports are complete. No robustness point criterion passed; no new search was opened.

Local tests: 35 passed. Ubuntu/Python 3.12 CI run [37621126425](https://github.com/numann44/engine-health-monitoring/actions/runs/37621126425) passed lint, tests, official-data audit and a small real-data training/prediction smoke. The first CI attempt exposed an unanchored Git ignore rule; the original data package files were restored without changing their pre-training hashes.

Release v0.1.0 is public at source `9cf2c08`. Both published assets were downloaded and SHA-256 verified. See [publication evidence](results/PUBLICATION_VERIFICATION.json), [artifact parity](results/ARTIFACT_VERIFICATION.json), and [CLI checks](results/CLI_VERIFICATION.json).

Streamlit deployment is being verified separately; local UI checks alone are not hosted evidence.
