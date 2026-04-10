import json
import queue as _stdlib_queue
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import torch
import torch.nn as nn
import transformers
import sentence_transformers
import asyncio
import csv
import requests
import sys


class _QueueLogHandler(logging.Handler):
    """Captures log records from a worker thread into a stdlib Queue for SSE forwarding."""

    def __init__(self, q: _stdlib_queue.Queue) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        self.q.put_nowait(self.format(record))

# Adjust imports to find src and config
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from config.paths import RUNS_DIR, TABLES_DIR, HF_CACHE_HINT_DIR, RESULTS_DIR, TEMP_DEMO_XAI_DIR
from config.experiment import get_embedding_config, XAI_CONFIG, PRIMARY_METRIC
from src.embeddings.factory import get_embedder
from src.xai.llm_explainer_v2 import NaturalLanguageExplainer
from src.xai.word_ablation_explainer import compute_contextual_ablation_keywords, resolve_guardrailed_verdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo_api")

HISTORY_FILE = RESULTS_DIR / "powertoy_history.json"
CACHE_FILE = RESULTS_DIR / "powertoy_cache.json"
DEMO_TEMP_DIR = TEMP_DEMO_XAI_DIR


def compute_leave_one_out_keywords(subject: str, embedder, model) -> list[dict]:
    def predict_fn(texts: list[str]) -> np.ndarray:
        embeddings = embedder.encode(texts)
        return model.predict_proba(embeddings)

    return compute_contextual_ablation_keywords(
        subject,
        predict_fn,
        max_features=6,
        min_share=0.04,
    )


def get_runtime_technologies() -> list[str]:
    technologies = ["Leave-One-Out XAI", "FastAPI"]

    embedding_name = _meta.get("embedding") if _meta else None
    if embedding_name:
        try:
            embedding_config = get_embedding_config(embedding_name)
            backend_label = {
                "sentence_transformer": "SentenceTransformers",
                "hf_encoder": "Transformers",
                "large_llm": "Transformers",
            }.get(embedding_config.get("type", ""), "Embeddings")
            if backend_label not in technologies:
                technologies.insert(1, backend_label)
        except Exception:
            pass

    llm_label = {
        "groq": "Groq",
        "hf": "HuggingFace Router",
        "local": "Transformers (Local Qwen)",
    }.get(getattr(llm_explainer, "engine", "local"), "LLM")
    if llm_label not in technologies:
        technologies.append(llm_label)

    return technologies

def get_model_status(repo_name: str):
    """Checks if a HuggingFace model is likely cached locally."""
    safe_repo = f"models--{repo_name.replace('/', '--')}"
    model_path = HF_CACHE_HINT_DIR / safe_repo
    if not model_path.exists():
        return "missing"
    # Check if there are blobs (actual data)
    blobs_path = model_path / "blobs"
    if not blobs_path.exists() or not any(blobs_path.iterdir()):
        return "incomplete"
    return "cached"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    load_models()
    yield
    # Clean up on shutdown if needed

app = FastAPI(title="Phishing XAI PowerToy", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/config")
async def get_config():
    global _meta, _hw_msg
    return {
        "embedding": _meta.get("embedding", "Desconocido") if _meta else "Cargando...",
        "classifier": _meta.get("classifier", "Desconocido") if _meta else "Cargando...",
        "hardware": _hw_msg,
        "run_id": _meta.get("run_id", "N/A") if _meta else "N/A",
        "technologies_used": get_runtime_technologies(),
    }

class AnalyzeRequest(BaseModel):
    subject: str

# ----------------- MODEL LOADING -----------------
def get_ranked_candidates():
    """Reads all_results.csv and returns a list of candidate models sorted by performance."""
    results_file = TABLES_DIR / "all_results.csv"
    if not results_file.exists():
        return []
    
    try:
        with open(results_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # Sort by PRIMARY_METRIC descending
            candidates = sorted(list(reader), key=lambda x: float(x.get(PRIMARY_METRIC, 0)), reverse=True)
            return candidates
    except Exception as e:
        logger.error(f"Error parseando resultados: {e}")
        return []

# Global variables to cache model in memory
_model = None
_embedder = None
_meta = None
_hw_msg = "Esperando detección de hardware..."
llm_explainer = NaturalLanguageExplainer(base_path=project_root)

from src.utils.hardware import HardwareManager

def load_models():
    global _model, _embedder, _meta, _hw_msg
    
    # 1. Hardware Awareness
    hw_manager = HardwareManager()
    
    # 2. Get historical results to find better alternatives if needed
    results_file = TABLES_DIR / "all_results.csv"
    results = []
    if results_file.exists():
        try:
            with open(results_file, "r", encoding="utf-8") as f:
                results = list(csv.DictReader(f))
        except Exception as e:
            logger.error(f"Error cargando base de resultados: {e}")

    # 2. Get all historical candidates
    candidates = get_ranked_candidates()
    
    # 3. Dynamic Selection Loop: Find the best that fits
    best_info = None
    suggestion = None
    
    if candidates:
        for cand in candidates:
            emb_name = cand["embedding"]
            config_sug = hw_manager.suggest_config(emb_name)
            if config_sug.get("can_run", True):
                best_info = cand
                suggestion = config_sug
                logger.info(f"🎯 Selección Adaptativa: {emb_name} es el mejor modelo compatible (F1: {cand.get(PRIMARY_METRIC)})")
                break
            else:
                logger.info(f"⏭️ Saltando {emb_name}: {config_sug['message']}")

    # 4. Fallback if no specific history fits or no history found
    if not best_info:
        logger.warning("No se encontró ningún modelo en el historial que se ajuste al hardware. Usando emergencia.")
        # If we have no suggestion yet, get stats for a very base model
        if not suggestion:
            suggestion = hw_manager.suggest_config("distilbert-base-uncased")

    # 6. Final Model Loading
    if best_info:
        run_id = best_info["run_id"]
        emb_name = best_info["embedding"]
        clf_name = best_info["classifier"]
        logger.info(f"🚀 Cargando: {emb_name} + {clf_name} (Run: {run_id})")
        
        run_dir = RUNS_DIR / run_id
        emb_safe = emb_name.replace("/", "__")
        model_path = run_dir / f"{emb_safe}__{clf_name}.joblib"
        if not model_path.exists(): model_path = run_dir / "best_model.joblib"
            
        _meta = {
            "embedding": emb_name,
            "classifier": clf_name,
            "run_id": run_id,
            "f1_macro": best_info.get("f1_macro", 0)
        }
    else:
        # Emergency last-resort fallback logic
        logger.info("Cargando el último run disponible...")
        latest_run = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "best_model.joblib").exists()], 
                           key=lambda x: x.stat().st_mtime)[-1]
        model_path = latest_run / "best_model.joblib"
        meta_path = latest_run / "best_model_meta.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            _meta = json.load(f)

    logger.info(f"Cargando pesos desde: {model_path}")
    _model = joblib.load(str(model_path))
    
    embedding_config = get_embedding_config(_meta['embedding'])
    _embedder = get_embedder(
        name=_meta['embedding'],
        model_config=embedding_config,
        device=torch.device(suggestion["device"]),
        precision=suggestion["dtype"],
        checkpoint_dir=RUNS_DIR 
    )
    
    precision_name = str(suggestion["dtype"]).split('.')[-1]
    _hw_msg = f"⚙️ {suggestion['device'].upper()} | {precision_name.upper()} | {_meta['embedding']}"
    if suggestion["status"] == "warning":
        _hw_msg = f"⚠️ FALLBACK: {suggestion['message']} ({precision_name.upper()})"

    logger.info(f"✅ Configuración finalizada: {_hw_msg}")


def save_to_history(result_data):
    """Saves analysis result to a local JSON file."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except:
            history = []
    
    import datetime
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **result_data
    }
    history.insert(0, entry)
    history = history[:50] # Limit to 50
    
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

_FAKE_BODY_PLACEHOLDER = "No se generó reconstrucción de cuerpo."
_POWERSAFE_CACHE_SCHEMA_VERSION = 3

def get_cached_result(subject: str, run_id: str):
    """Retrieves a cached analysis if it exists for the same subject and model.
    Entries with a placeholder fake_body are ignored so the body gets regenerated."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
            subject_clean = subject.strip().lower()
            for entry in cache:
                if entry.get("subject", "").strip().lower() == subject_clean and entry.get("run_id") == run_id:
                    if entry.get("cache_schema_version", 0) < _POWERSAFE_CACHE_SCHEMA_VERSION:
                        return None
                    if entry.get("fake_body", "") == _FAKE_BODY_PLACEHOLDER:
                        return None  # force regeneration with the new body engine
                    # force regeneration if keywords lack the 'positive' direction field
                    kws = entry.get("keywords", [])
                    if kws and "positive" not in kws[0]:
                        return None
                    return entry
    except Exception as e:
        logger.error(f"Error reading cache: {e}")
    return None

def save_to_cache(result_data: dict, run_id: str):
    """Saves an analysis result to the local cache."""
    cache = []
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except:
            cache = []
    
    # Check if entry already exists to avoid duplicates
    subject_clean = result_data.get("subject", "").strip().lower()
    cache = [e for e in cache if not (e.get("subject", "").strip().lower() == subject_clean and e.get("run_id") == run_id)]
    
    # Add new entry with run_id
    entry = {**result_data, "run_id": run_id, "cache_schema_version": _POWERSAFE_CACHE_SCHEMA_VERSION}
    cache.insert(0, entry)
    cache = cache[:200]  # Limit cache size
    
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# ----------------- INFERENCE ENDPOINT (STREAMING) -----------------
@app.post("/api/analyze")
async def analyze_subject(req: AnalyzeRequest):
    global _model, _embedder, _meta
    if not _model or not _embedder:
        raise HTTPException(status_code=500, detail="Models not loaded")
        
    subject = req.subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="Subject is empty")

    async def event_generator():
        # --- CACHE CHECK ---
        run_id = _meta.get("run_id", "unknown")
        cached = get_cached_result(subject, run_id)
        if cached:
            yield json.dumps({"type": "log", "content": "♻️ Resultados recuperados de la caché forense local."}) + "\n"
            yield json.dumps({"type": "log", "content": "✅ Análisis previo detectado: Restaurando interpretatibilidad..."}) + "\n"
            yield json.dumps({"type": "result", "data": cached}) + "\n"
            return
            
        # 0. Check Model Status
        repo = _meta.get("embedding", "")
        status = get_model_status(repo)
        
        if status == "missing" or status == "incomplete":
            yield json.dumps({"type": "log", "content": f"📡 AVISO: Modelo '{repo}' no detectado en caché local."}) + "\n"
            yield json.dumps({"type": "log", "content": "Iniciando descarga técnica desde HuggingFace Hub (esto puede tardar unos minutos)..."}) + "\n"
        else:
            yield json.dumps({"type": "log", "content": "🔍 Verificando integridad del modelo en caché local... OK"}) + "\n"

        # 1. Technical Details
        global _hw_msg
        model_name = repo or "Desconocido"
        classifier_name = _meta.get("classifier", "Desconocido")
        params_str = "Default"
        
        yield json.dumps({"type": "log", "content": f"💻 Sistema Adaptativo: {_hw_msg}"}) + "\n"
        if hasattr(_model, "get_params"):
            p = _model.get_params()
            relevant_keys = ["C", "kernel", "n_estimators", "max_depth", "alpha", "solver", "n_neighbors"]
            relevant = {k: v for k, v in p.items() if k in relevant_keys}
            if relevant:
                params_str = ", ".join([f"{k}={v}" for k, v in relevant.items()])

        yield json.dumps({"type": "log", "content": f"⚙️ Configuración Forense: {model_name} + {classifier_name} ({params_str})"}) + "\n"
        yield json.dumps({"type": "log", "content": "Iniciando análisis semántico de seguridad..."}) + "\n"
        
        # 2. Embedding & Prediction (NON-BLOCKING)
        yield json.dumps({"type": "log", "content": "🤖 Calculando representación semántica (cargando modelo en RAM, puede tardar 5-10s)..."}) + "\n"
        
        # We use asyncio.to_thread to avoid blocking the main event loop
        emb = await asyncio.to_thread(_embedder.encode, [subject])
        probs = await asyncio.to_thread(_model.predict_proba, emb)
        probs = probs[0]
        
        phishing_probability = float(probs[1])
        is_phishing = bool(phishing_probability > 0.5)
        confidence = float(probs[1] if is_phishing else probs[0])
        yield json.dumps({"type": "log", "content": f"✅ Procesamiento completado (Confianza: {confidence*100:.1f}%)"}) + "\n"
        
        if is_phishing:
            yield json.dumps({"type": "log", "content": "🚨 ¡Alerta detectada! Activando protocolos de explicabilidad XAI..."}) + "\n"
        else:
            yield json.dumps({"type": "log", "content": "🛡️ Correo benigno detectado. Ejecutando explicabilidad completa para validación forense..."}) + "\n"

        yield json.dumps({"type": "log", "content": "🧠 Ejecutando atribución leave-one-out por palabra..."}) + "\n"
        lime_keywords = await asyncio.to_thread(compute_leave_one_out_keywords, subject, _embedder, _model)
        yield json.dumps({"type": "log", "content": f"✅ Atribución unificada completada: {len(lime_keywords)} palabras relevantes."}) + "\n"
        verdict = resolve_guardrailed_verdict(phishing_probability, lime_keywords)
        if verdict["decision_source"] == "attribution_guardrail" and not is_phishing:
            yield json.dumps({"type": "log", "content": "⚠️ Guardrail forense activado: la atribución contradice un veredicto benigno fronterizo y eleva el caso a phishing."}) + "\n"
        phishing_probability = float(verdict["probability"])
        is_phishing = bool(verdict["is_phishing"])
        confidence = float(phishing_probability if is_phishing else (1.0 - phishing_probability))
        yield json.dumps({"type": "log", "content": "📜 Generando razonamiento maestro alineado con la atribución científica..."}) + "\n"
        explanation = await asyncio.to_thread(llm_explainer.generate_explanation, subject, phishing_probability, lime_keywords)
        yield json.dumps({"type": "log", "content": "🎭 Generando cuerpo del mensaje en el idioma del asunto..."}) + "\n"
        fake_body = await asyncio.to_thread(llm_explainer.generate_email_body, subject, is_phishing)

        final_data = {
            "status": "phishing" if is_phishing else "safe",
            "subject": subject,
            "confidence": round(confidence * 100, 1),
            "keywords": lime_keywords,
            "explanation": explanation,
            "fake_body": fake_body
        }
        
        # --- SAVE TO CACHE & HISTORY ---
        save_to_cache(final_data, run_id)
        save_to_history(final_data)
        
        yield json.dumps({"type": "result", "data": final_data}) + "\n"
        yield json.dumps({"type": "log", "content": "💾 Análisis finalizado y guardado exitosamente en la caché forense."}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")

# Serve Frontend
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
