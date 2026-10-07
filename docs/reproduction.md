# Reproduction

Use Python 3.12 and `requirements.lock`. Install the project with `pip install --no-deps -e .`. Commands run from the repository root.

```bash
engine-health download
engine-health audit
engine-health prepare
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
pytest -q
python scripts/smoke.py
engine-health pilot
# Commit code, dependencies and splits before freezing.
engine-health freeze
engine-health study --full
```

The full command completes the declared fits, freezes all selections, evaluates the official tests, generates reports and packages model assets. It does not push to GitHub or publish a website. It does not add candidates after a failed hypothesis.

If interrupted, use the same command or `engine-health resume --full` in the **original environment**. Existing compatible completed runs are checked and reused; incomplete neural runs resume their optimizer, RNG and epoch state. A changed source/environment/configuration fails closed. Never delete a lock file held by a live process.

`outputs/study-v1/status.json` identifies the current phase/job. Each neural run includes `progress.json`, `history.json`, `last.pt`, `selected.pt` and `summary.json`. The selected file is not necessarily from the final epoch.

The short pilot uses real training windows to compare CPU/MPS with disposable weights; it neither evaluates test data nor chooses model parameters. Training is single-threaded for controlled timings and reproducibility. Measured epoch times, not synthetic benchmark extrapolation alone, determine the remaining-time estimate.

```bash
engine-health predict --subset FD001 --csv /path/to/measurements.csv --unit 1
streamlit run app/streamlit_app.py
```

The demo becomes model-enabled when `assets/model-registry.json` and its local or published bundle are available. Source-only installations show a clear pending-release state instead of fabricated predictions.

## Exact original study source

The original training/evaluation revision is preserved in the checksum-listed `study-v1-frozen-source.tar.gz` release asset. Extract it into a separate directory to reproduce the exact declared source identity. Later release changes add portable CLI prediction, a CPU inference benchmark and family-consistent chart colors; they do not change fitted preprocessing, model mathematics, selected checkpoints or test predictions. The declaration records the original local commit and source hashes, rather than asserting that a later publication commit trained the models.

For a downloaded release, `engine-health predict` loads the checksum-pinned bundle automatically. `engine-health benchmark --subset FD001 --csv /path/to/measurements.csv --unit 1` measures 30 warm end-to-end CPU inferences; `pilot` remains the distinct training-device comparison command. Use `--family`, `--cycle` and `--condition` to inspect fixed alternatives.
