# Free deployment and release checks

Repository: `numann44/engine-health-monitoring`. Entrypoint: `app/streamlit_app.py`. Runtime: **Python 3.12**, CPU. A working public URL is recorded only after a real deployment.

Community Cloud discovers the root `requirements.txt`, which includes a separate deployment lock and selects the checksum-pinned official PyTorch CPU wheel on Linux x86_64. This avoids installing unused CUDA libraries and avoids giving a secondary package index precedence over PyPI. The training lock is preserved unchanged. Community Cloud replaced PyArrow 25.0.1 because of its known server compatibility issue; deployment explicitly pins 24.0.0. Numerical package versions and saved models are unchanged. The app imports the repository's `src` package directly; no shell bootstrap, secrets, database, REST server or cloud training job is required. The [Streamlit dependency documentation](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) explains dependency-file discovery and Python-version alignment.

## Model assets

Publish `outputs/release-v0.1.0/engine-health-v0.1.0.zip` as a GitHub release asset at the URL in `assets/model-registry.json`. The manifest pins the whole archive and the internal registry. Every model and preprocessor has its own SHA-256. The app downloads and verifies the bundle once, then caches models in memory. It obtains example measurements from NASA separately. It must report unavailable resources rather than inventing results.

## Local acceptance

After all fixed fits and evaluation have completed:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/verify_release.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/presentation.py
python scripts/uncertainty_summary.py
python scripts/social_preview.py
streamlit run app/streamlit_app.py
```

Check each scenario and model, short history, slider changes, the four downloadable outputs, real-source CSV upload, missing sensor cells and invalid cycles. Inspect exported values against the common inference engine. Test desktop and narrow browser widths; browser resizing is not a physical phone test.

## Hosted acceptance

After GitHub Linux CI and asset publication, deploy the public repository through [Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app). Explicitly select Python 3.12. Record the hosted URL, source commit, asset hash and actual browser checks. The hosted example, uploaded version of that example and downloaded JSON must match within documented floating-point tolerance. Inspect CSV rows and PNG content, stress controls and error behavior. A passing local test does not stand in for a hosted test.

The release remains experimental simulation-data research even if every software check passes. Neither a success badge nor a stable stress response establishes real-aircraft reliability.

## Published instance

[Engine Health Monitoring](https://numan-engine-health-monitoring.streamlit.app) runs from `main` on free Community Cloud CPU. See [actual hosted verification](results/DEPLOYMENT_VERIFICATION.json) for example/upload parity, missing and invalid data, exports, all four selected defaults, and browser-test limitations.
