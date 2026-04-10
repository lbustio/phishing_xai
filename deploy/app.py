from __future__ import annotations

import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from config.experiment import get_embedding_config
from src.embeddings.factory import get_embedder
from src.xai.word_ablation_explainer import (
    compute_contextual_ablation_keywords,
    resolve_guardrailed_verdict,
)


logger = logging.getLogger("deploy_demo")
logging.basicConfig(level=logging.INFO)


class AnalyzeRequest(BaseModel):
    subject: str


_model = None
_embedder = None
_bundle = None


def detect_language(text: str) -> str:
    try:
        from langdetect import detect

        return detect((text or "").strip()).lower()
    except Exception:
        lowered = (text or "").lower()
        if re.search(r"[áéíóúñ¿¡]", lowered):
            return "es"
        if re.search(r"\b(le|la|de|bonjour|merci|votre|compte)\b", lowered):
            return "fr"
        return "en"


def generate_template_explanation(subject: str, phishing_probability: float, keywords: list[dict]) -> str:
    verdict = "phishing" if phishing_probability >= 0.5 else "legítimo"
    probability_pct = phishing_probability * 100.0

    if keywords:
        top = keywords[:3]
        feature_text = ", ".join(f'"{item["word"]}" ({item["impact"]}%)' for item in top)
        direction = "elevan el riesgo" if phishing_probability >= 0.5 else "reducen el riesgo"
        return (
            f"El modelo clasifica este asunto como {verdict} con una probabilidad de {probability_pct:.1f}%. "
            f"Las señales textuales más influyentes en esta versión simplificada del demo son {feature_text}. "
            f"En conjunto, esas expresiones {direction} según la atribución leave-one-out calculada sobre el asunto. "
            "Esta explicación resume el comportamiento del pipeline desplegado y no debe interpretarse como una garantía causal absoluta."
        )

    return (
        f"El modelo clasifica este asunto como {verdict} con una probabilidad de {probability_pct:.1f}%. "
        "En este caso no se aislaron palabras individuales suficientemente dominantes, por lo que la señal parece depender más del patrón global del embedding. "
        "La explicación debe leerse como evidencia diagnóstica del modelo desplegado, no como una prueba definitiva sobre el mensaje real."
    )


def generate_template_body(subject: str, is_phishing: bool) -> str:
    language = detect_language(subject)
    if language == "es":
        return (
            "Hemos detectado actividad inusual en su cuenta y necesitamos que revise la información registrada para evitar una interrupción del servicio."
            if is_phishing
            else "Le escribimos para confirmar la información relacionada con su solicitud y darle seguimiento al trámite correspondiente."
        )
    if language == "fr":
        return (
            "Nous avons détecté une activité inhabituelle sur votre compte et nous avons besoin d'une vérification rapide pour éviter une interruption du service."
            if is_phishing
            else "Nous vous contactons afin de confirmer les informations liées à votre demande et assurer le suivi normal du dossier."
        )
    return (
        "We detected unusual activity on your account and need a quick verification to avoid a service interruption."
        if is_phishing
        else "We are contacting you to confirm the information related to your request and continue the normal follow-up process."
    )


def compute_keywords(subject: str) -> list[dict]:
    def predict_fn(texts: list[str]) -> np.ndarray:
        embeddings = _embedder.encode(texts)
        return _model.predict_proba(embeddings)

    return compute_contextual_ablation_keywords(
        subject,
        predict_fn,
        max_features=6,
        min_share=0.04,
    )


def load_bundle() -> None:
    global _model, _embedder, _bundle

    bundle_dir = Path(os.environ.get("DEPLOY_BUNDLE_DIR", PROJECT_ROOT / "deploy" / "bundle"))
    bundle_manifest_path = bundle_dir / "deploy_bundle.json"
    if not bundle_manifest_path.exists():
        raise FileNotFoundError(
            f"Deploy bundle not found at {bundle_manifest_path}. Run 'python deploy/prepare_bundle.py' first."
        )

    _bundle = json.loads(bundle_manifest_path.read_text(encoding="utf-8"))
    model_path = bundle_dir / _bundle["model_path"]
    _model = joblib.load(model_path)

    embedding_name = _bundle["embedding"]
    embedding_config = get_embedding_config(embedding_name)
    _embedder = get_embedder(
        name=embedding_name,
        model_config=embedding_config,
        device=torch.device("cpu"),
        precision=torch.float32,
        checkpoint_dir=PROJECT_ROOT / "results" / "runs",
    )

    logger.info(
        "Deploy demo loaded: embedding=%s classifier=%s run_id=%s",
        _bundle["embedding"],
        _bundle["classifier"],
        _bundle["run_id"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_bundle()
    yield


app = FastAPI(title="Phishing XAI Deploy Demo", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/config")
async def get_config():
    return {
        "embedding": _bundle["embedding"],
        "classifier": _bundle["classifier"],
        "run_id": _bundle["run_id"],
        "mode": "simplified_public_demo",
        "limitations": [
            "single preselected model",
            "no dataset-dependent semantic scatter",
            "no persistent history view",
            "template-based body generation",
        ],
    }


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.post("/api/analyze")
async def analyze_subject(req: AnalyzeRequest):
    subject = req.subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="Subject is empty")

    embedding = _embedder.encode([subject])
    probs = _model.predict_proba(embedding)[0]
    phishing_probability = float(probs[1])

    keywords = compute_keywords(subject)
    verdict = resolve_guardrailed_verdict(phishing_probability, keywords)
    phishing_probability = float(verdict["probability"])
    is_phishing = bool(verdict["is_phishing"])
    confidence = float(phishing_probability if is_phishing else (1.0 - phishing_probability))

    explanation = generate_template_explanation(subject, phishing_probability, keywords)
    fake_body = generate_template_body(subject, is_phishing)

    return {
        "status": "phishing" if is_phishing else "safe",
        "confidence": round(confidence * 100, 1),
        "subject": subject,
        "keywords": keywords,
        "explanation": explanation,
        "fake_body": fake_body,
        "mode": "simplified_public_demo",
    }


static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
