# Deploy Demo for Render

This folder contains everything needed to deploy a simplified public demo of `phishing_xai` on Render.

The deployable demo is intentionally narrower than the local PowerToy:

- it serves a lightweight FastAPI app
- it uses one preselected trained model only
- it keeps word-level attribution
- it keeps the Spanish reasoning text
- it keeps the synthetic email body in the detected language of the subject
- it does not expose the full semantic scatter, history view, PDF export, or dataset-dependent artifacts

The simplification is deliberate for two reasons:

1. Free hosting has tighter CPU, RAM, startup-time, and disk constraints.
2. The full semantic scatter depends on private dataset artifacts that should not be exposed in a public demo.

## Files

- `app.py`: simplified FastAPI application for Render
- `prepare_bundle.py`: copies one selected trained model into `deploy/bundle/`
- `requirements.txt`: Python dependencies for the deployed demo
- `render.yaml`: Render service template
- `static/`: lightweight frontend
- `bundle/`: sanitized deploy bundle with the public-demo model artifacts

## Default Deployment Profile

The default recommended public-demo profile is:

- run id: `20260328_031334`
- embedding: `distilbert-base-uncased`
- classifier: `svm_rbf`

This was chosen because it is materially lighter than the larger embedding models while still performing well in your results.

## Step 1. Prepare the Bundle

From the repository root:

```powershell
python deploy/prepare_bundle.py
```

This creates:

```text
deploy/bundle/
|-- model.joblib
|-- deploy_bundle.json
|-- model_metadata.json
`-- run_manifest.json
```

You can override the default selection:

```powershell
python deploy/prepare_bundle.py --run-id 20260328_031334 --embedding distilbert-base-uncased --classifier svm_rbf
```

## Step 2. Deploy to Render

You can deploy it either:

- manually as a Render Web Service
- or by copying `deploy/render.yaml` to the repository root as `render.yaml` and using a Render Blueprint

### Manual Render settings

- Environment: `Python`
- Build Command:

```text
pip install -r deploy/requirements.txt
```

- Start Command:

```text
uvicorn deploy.app:app --host 0.0.0.0 --port $PORT
```

## Step 3. Optional Environment Variables

- `HF_TOKEN`: optional, useful if Hugging Face rate limits become a problem
- `DEPLOY_BUNDLE_DIR`: optional; defaults to `deploy/bundle`

## Important Notes

- the committed bundle must remain sanitized and must not expose the real dataset filename or local host paths
- the public demo should be treated as a lightweight showcase, not as the full research console
- if you change the selected model, rerun `prepare_bundle.py` before deploying and review the generated bundle before committing it
