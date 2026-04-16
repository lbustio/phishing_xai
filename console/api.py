import json
import base64
import queue as _stdlib_queue
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import joblib
import numpy as np
import torch
import asyncio
import csv
import sys
from sklearn.decomposition import PCA


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

from config.paths import RUNS_DIR, TABLES_DIR, HF_CACHE_HINT_DIR, RESULTS_DIR, DATA_DIR, FRONTEND_ANALYSES_DIR
from config.experiment import get_embedding_config, PRIMARY_METRIC
from src.data_loader import load_dataset
from src.embedding_store import EmbeddingStore
from src.embeddings.cache_utils import get_hf_model_cache_status
from src.embeddings.factory import get_embedder
from src.xai.llm_explainer_v2 import NaturalLanguageExplainer
from src.xai.word_ablation_explainer import compute_contextual_ablation_keywords, resolve_guardrailed_verdict, compute_counterfactual_flip

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("xai_analyzer")

HISTORY_FILE = RESULTS_DIR / "xai_analyzer_history.json"
CACHE_FILE = RESULTS_DIR / "xai_analyzer_cache.json"


_FRENCH_CHARS  = set("àâçèêëîïôùûœæ")
_PORTUGUESE_CHARS = set("ãõ")
_GERMAN_CHARS  = set("äöß")
_SPANISH_OK    = set("ñáéíóúü¡¿")

def detect_language_warning(subject: str) -> str | None:
    chars = set(subject.lower())
    if chars & _GERMAN_CHARS:
        return "Posible alemán — fuera de la distribución de entrenamiento (inglés/español). La confianza puede estar sobreestimada."
    if chars & _PORTUGUESE_CHARS:
        return "Posible portugués — fuera de la distribución de entrenamiento (inglés/español). La confianza puede estar sobreestimada."
    if (chars & _FRENCH_CHARS) and not (chars & _SPANISH_OK):
        return "Posible francés — fuera de la distribución de entrenamiento (inglés/español). La confianza puede estar sobreestimada."
    return None

def _compute_counterfactual(subject: str, base_prob: float) -> dict | None:
    def predict_fn(texts: list[str]) -> np.ndarray:
        return _model.predict_proba(_embedder.encode(texts))
    return compute_counterfactual_flip(subject, predict_fn, base_prob)

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
    return get_hf_model_cache_status(repo_name, HF_CACHE_HINT_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    # Pre-warm embedder and semantic cache so the first analysis request is fast
    try:
        logger.info("Pre-calentando embedder...")
        _embedder.encode(["warmup"])
        logger.info("Pre-cargando caché semántica (dataset + PCA)...")
        _prepare_semantic_cache()
        logger.info("✅ Pre-calentamiento completo. Listo para recibir requests.")
    except Exception as e:
        logger.warning(f"Pre-calentamiento falló (no crítico): {e}")
    yield

app = FastAPI(title="Phishing XAI Analyzer", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/config")
async def get_config():
    global _meta, _hw_msg

    if not _meta:
        return {
            "embedding": "Cargando...",
            "classifier": "Cargando...",
            "hardware": _hw_msg,
            "run_id": "N/A",
            "technologies_used": [],
        }

    embedding_name = _meta.get("embedding", "")
    classifier_name = _meta.get("classifier", "")
    run_id = _meta.get("run_id", "N/A")

    # Embedding config & rationale
    emb_config = {}
    try:
        emb_config = get_embedding_config(embedding_name)
    except Exception:
        pass

    # Classifier note & class name
    from config.experiment import CLASSIFIERS_GRID
    clf_entry = CLASSIFIERS_GRID.get(classifier_name, {})
    clf_note = clf_entry.get("note", "")
    clf_class = clf_entry.get("class_name", classifier_name)

    # Classifier .joblib size on disk
    model_size_mb = None
    try:
        run_dir = RUNS_DIR / run_id
        emb_safe = embedding_name.replace("/", "__")
        model_path = run_dir / f"{emb_safe}__{classifier_name}.joblib"
        if not model_path.exists():
            model_path = run_dir / "best_model.joblib"
        if model_path.exists():
            model_size_mb = round(model_path.stat().st_size / (1024 * 1024), 2)
    except Exception:
        pass

    # HF cache size & download date
    hf_cache_size_mb = None
    hf_download_date = None
    try:
        hf_model_slug = embedding_name.replace("/", "--")
        hf_cache_dir = HF_CACHE_HINT_DIR / f"models--{hf_model_slug}"
        if hf_cache_dir.exists():
            total_bytes = sum(f.stat().st_size for f in hf_cache_dir.rglob("*") if f.is_file())
            hf_cache_size_mb = round(total_bytes / (1024 * 1024), 1)
            hf_download_date = datetime.fromtimestamp(hf_cache_dir.stat().st_mtime).strftime("%Y-%m-%d")
    except Exception:
        pass

    # Run manifest: training date & device
    run_created_at = None
    run_device = None
    try:
        manifest = _load_run_manifest(run_id)
        run_created_at = manifest.get("created_at")
        run_device = manifest.get("device")
    except Exception:
        pass

    # All ranked candidates + active candidate metrics from CSV
    candidates = get_ranked_candidates()
    active_metrics = None
    top_candidates = []
    for i, cand in enumerate(candidates):
        is_active = (cand.get("embedding") == embedding_name and cand.get("classifier") == classifier_name)
        row = {
            "rank": i + 1,
            "embedding": cand.get("embedding"),
            "classifier": cand.get("classifier"),
            "run_id": cand.get("run_id"),
            "f1_macro": round(float(cand.get("f1_macro", 0)), 4),
            "accuracy": round(float(cand.get("accuracy", 0)), 4),
            "roc_auc": round(float(cand.get("roc_auc", 0)), 4),
            "active": is_active,
        }
        top_candidates.append(row)
        if is_active:
            try:
                best_params = json.loads(cand.get("best_params", "{}"))
            except Exception:
                best_params = {}
            active_metrics = {
                "f1_macro": round(float(cand.get("f1_macro", 0)), 4),
                "f1_macro_std": round(float(cand.get("f1_macro_std", 0)), 4),
                "f1_weighted": round(float(cand.get("f1_weighted", 0)), 4),
                "accuracy": round(float(cand.get("accuracy", 0)), 4),
                "precision_macro": round(float(cand.get("precision_macro", 0)), 4),
                "recall_macro": round(float(cand.get("recall_macro", 0)), 4),
                "roc_auc": round(float(cand.get("roc_auc", 0)), 4),
                "best_params": best_params,
                "timestamp": cand.get("timestamp"),
            }

    selection_reason = None
    if active_metrics:
        selection_reason = (
            f"Seleccionado automáticamente por obtener el mayor F1-macro "
            f"({active_metrics['f1_macro']:.4f} ± {active_metrics['f1_macro_std']:.4f}) "
            f"entre todas las combinaciones de embedder + clasificador evaluadas "
            f"en el experimento, con validación cruzada de 5 folds."
        )

    return {
        "embedding": embedding_name,
        "classifier": classifier_name,
        "hardware": _hw_msg,
        "run_id": run_id,
        "technologies_used": get_runtime_technologies(),
        "embedding_details": {
            "paradigm": emb_config.get("paradigm"),
            "type": emb_config.get("type"),
            "rationale": emb_config.get("rationale"),
            "batch_size": emb_config.get("batch_size"),
            "hf_url": f"https://huggingface.co/{embedding_name}" if "/" in embedding_name else None,
            "hf_cache_size_mb": hf_cache_size_mb,
            "hf_download_date": hf_download_date,
        },
        "classifier_details": {
            "class_name": clf_class,
            "note": clf_note,
            "best_params": active_metrics.get("best_params") if active_metrics else {},
        },
        "metrics": active_metrics,
        "selection_reason": selection_reason,
        "top_candidates": top_candidates[:10],
        "run_created_at": run_created_at,
        "run_device": run_device,
        "model_size_mb": model_size_mb,
    }

class AnalyzeRequest(BaseModel):
    subject: str
    email_body: str | None = None

class FeedbackRequest(BaseModel):
    feedback: str  # "correct" or "incorrect"


class SemanticRequest(BaseModel):
    subject: str
    status: str | None = None


class FrontendAssetRequest(BaseModel):
    analysis_id: str
    scatter_png_base64: str | None = None
    pdf_base64: str | None = None


def _sanitize_slug(text: str, limit: int = 48) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(text))
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return (cleaned or "analysis")[:limit].strip("-") or "analysis"


def _create_frontend_analysis_dir(subject: str) -> tuple[str, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = f"{timestamp}_{_sanitize_slug(subject)}"
    analysis_dir = FRONTEND_ANALYSES_DIR / analysis_id
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return analysis_id, analysis_dir


def _persist_frontend_analysis_artifacts(
    analysis_id: str,
    analysis_dir: Path,
    result_data: dict,
    log_lines: list[str],
    run_id: str,
) -> None:
    payload = _to_jsonable(result_data)
    semantic_map = payload.get("semantic_map")
    metadata = {
        "analysis_id": analysis_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "embedding": _meta.get("embedding") if _meta else None,
        "classifier": _meta.get("classifier") if _meta else None,
        "status": payload.get("status"),
        "confidence": payload.get("confidence"),
        "subject": payload.get("subject"),
        "artifacts": {
            "result_json": "analysis_result.json",
            "log_txt": "processing_log.txt",
            "reasoning_txt": "reasoning.txt",
            "keywords_json": "keywords.json",
            "body_txt": "synthetic_email_body.txt",
            "semantic_map_json": "semantic_map.json" if semantic_map else None,
            "scatter_png": "semantic_scatter.png",
            "report_pdf": "forensic_report.pdf",
        },
    }

    (analysis_dir / "analysis_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (analysis_dir / "processing_log.txt").write_text("\n".join(log_lines).strip() + "\n", encoding="utf-8")
    (analysis_dir / "reasoning.txt").write_text(str(payload.get("explanation") or ""), encoding="utf-8")
    (analysis_dir / "keywords.json").write_text(
        json.dumps(payload.get("keywords", []), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (analysis_dir / "synthetic_email_body.txt").write_text(str(payload.get("fake_body") or ""), encoding="utf-8")
    if semantic_map:
        (analysis_dir / "semantic_map.json").write_text(
            json.dumps(semantic_map, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    (analysis_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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


def save_to_history(result_data, embedding: np.ndarray | None = None):
    """Saves analysis result to a local JSON file."""
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    import datetime
    persisted = {k: v for k, v in result_data.items() if k != "semantic_map"}
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **persisted,
    }
    if embedding is not None:
        entry["_embedding"] = embedding.tolist()
    history.insert(0, entry)
    history = history[:50]

    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return 0  # newest entry is always index 0

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

@app.delete("/api/history")
async def clear_history():
    if HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")
    return {"ok": True}

@app.delete("/api/history/{index}")
async def delete_history_entry(index: int):
    if not HISTORY_FILE.exists():
        raise HTTPException(status_code=404, detail="No history file found")
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading history")
    if index < 0 or index >= len(history):
        raise HTTPException(status_code=404, detail="History entry not found")
    history.pop(index)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return {"ok": True}

@app.get("/api/stats")
async def get_stats():
    import datetime
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    today = datetime.date.today().isoformat()
    total = len(history)
    today_count = sum(1 for h in history if h.get("timestamp", "").startswith(today))
    phishing_count = sum(1 for h in history if h.get("is_phishing", h.get("status") == "phishing"))
    avg_conf = round(sum(h.get("confidence", 0) for h in history) / total, 1) if total else 0
    buckets = {"0-20": 0, "20-40": 0, "40-60": 0, "60-80": 0, "80-100": 0}
    for h in history:
        c = h.get("confidence", 0)
        if c < 20: buckets["0-20"] += 1
        elif c < 40: buckets["20-40"] += 1
        elif c < 60: buckets["40-60"] += 1
        elif c < 80: buckets["60-80"] += 1
        else: buckets["80-100"] += 1
    return {
        "total": total, "today": today_count,
        "phishing": phishing_count, "legitimate": total - phishing_count,
        "phishing_rate": round(phishing_count / total * 100, 1) if total else 0,
        "avg_confidence": avg_conf, "buckets": buckets,
        "feedback_correct": sum(1 for h in history if h.get("feedback") == "correct"),
        "feedback_incorrect": sum(1 for h in history if h.get("feedback") == "incorrect"),
    }

@app.post("/api/feedback/{index}")
async def submit_feedback(index: int, req: FeedbackRequest):
    if req.feedback not in ("correct", "incorrect"):
        raise HTTPException(status_code=400, detail="feedback must be 'correct' or 'incorrect'")
    if not HISTORY_FILE.exists():
        raise HTTPException(status_code=404, detail="No history")
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading history")
    if index < 0 or index >= len(history):
        raise HTTPException(status_code=404, detail="Entry not found")
    history[index]["feedback"] = req.feedback
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    return {"ok": True}

class SimilarRequest(BaseModel):
    subject: str

@app.post("/api/similar")
async def get_similar(req: SimilarRequest):
    if not _embedder:
        return []
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            pass
    query_lower = req.subject.strip().lower()
    candidates = [h for h in history if h.get("subject", "").strip().lower() != query_lower and "_embedding" in h]
    if not candidates:
        return []
    query_emb = await asyncio.to_thread(_embedder.encode, [req.subject])
    query_norm = _normalize_rows(np.asarray(query_emb, dtype=np.float32))
    cand_embs = np.asarray([h["_embedding"] for h in candidates], dtype=np.float32)
    cand_norm = _normalize_rows(cand_embs)
    sims = (cand_norm @ query_norm.T).flatten()
    top_idx = np.argsort(sims)[::-1][:3]
    result = []
    for idx in top_idx:
        sim = float(sims[idx])
        if sim < 0.25:
            continue
        h = candidates[idx]
        result.append({
            "subject": h.get("subject", ""),
            "is_phishing": h.get("is_phishing", h.get("status") == "phishing"),
            "confidence": h.get("confidence", 0),
            "timestamp": h.get("timestamp", ""),
            "similarity": round(sim * 100, 1),
            "feedback": h.get("feedback"),
        })
    return result

_FAKE_BODY_PLACEHOLDER = "No se generó reconstrucción de cuerpo."
_POWERSAFE_CACHE_SCHEMA_VERSION = 4
_semantic_cache = None

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
        except Exception:
            cache = []
    
    # Check if entry already exists to avoid duplicates
    subject_clean = result_data.get("subject", "").strip().lower()
    cache = [e for e in cache if not (e.get("subject", "").strip().lower() == subject_clean and e.get("run_id") == run_id)]
    
    # Add new entry with run_id
    persisted = {k: v for k, v in result_data.items() if k != "semantic_map"}
    entry = {**persisted, "run_id": run_id, "cache_schema_version": _POWERSAFE_CACHE_SCHEMA_VERSION}
    cache.insert(0, entry)
    cache = cache[:200]  # Limit cache size
    
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")


def _label_name(label_value: int) -> str:
    return "phishing" if int(label_value) == 1 else "legitimate"


def _truncate_text(text: str, limit: int = 160) -> str:
    cleaned = str(text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return matrix / norms


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _load_run_manifest(run_id: str) -> dict:
    manifest_path = RUNS_DIR / run_id / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No existe el manifiesto del run '{run_id}'.")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_dataset_path(data_path_raw: str) -> Path:
    candidate = Path(data_path_raw)
    if candidate.exists():
        return candidate

    filename = candidate.name
    local_candidate = DATA_DIR / filename
    if local_candidate.exists():
        logger.info(
            "La ruta historica del dataset no existe en este entorno. Se reutilizara la copia local '%s'.",
            local_candidate,
        )
        return local_candidate

    matches = sorted(DATA_DIR.glob(f"*{filename}*"))
    if matches:
        logger.info(
            "La ruta historica del dataset no existe en este entorno. Se reutilizara la coincidencia local '%s'.",
            matches[0],
        )
        return matches[0]

    raise FileNotFoundError(
        f"No existe el dataset en {candidate}. Tampoco se encontro una copia local equivalente en '{DATA_DIR}'."
    )


def _prepare_semantic_cache() -> dict | None:
    global _semantic_cache, _meta, _embedder
    if not _meta or not _embedder:
        logger.warning("No se puede preparar la cache semantica porque el modelo o el embedder no estan cargados.")
        return None

    run_id = _meta.get("run_id")
    embedding_name = _meta.get("embedding")
    if not run_id or not embedding_name:
        logger.warning("No se puede preparar la cache semantica porque faltan run_id o embedding en _meta.")
        return None

    cache_key = (run_id, embedding_name)
    if _semantic_cache and _semantic_cache.get("cache_key") == cache_key:
        logger.info("Mapa semantico: reutilizando cache en memoria para run=%s embedding=%s.", run_id, embedding_name)
        return _semantic_cache

    logger.info("Mapa semantico: cargando manifiesto del run '%s'.", run_id)
    manifest = _load_run_manifest(run_id)
    data_path = _resolve_dataset_path(manifest["data_path"])
    logger.info("Mapa semantico: dataset resuelto en '%s'.", data_path)
    dataset = load_dataset(data_path)
    dataset_fingerprint = manifest.get("dataset_fingerprint") or dataset.dataset_fingerprint
    logger.info(
        "Mapa semantico: dataset preparado con %s filas y fingerprint %s.",
        dataset.size,
        dataset_fingerprint,
    )

    store = EmbeddingStore(dataset_fingerprint)
    embeddings = store.load(embedding_name)
    if embeddings is None or len(embeddings) != dataset.size:
        logger.info("No se encontro cache de embeddings reutilizable para el mapa semantico; se calcularan en vivo.")
        embeddings = _embedder.encode(dataset.texts)
        store.save(
            embedding_name,
            embeddings,
            {
                "dataset_fingerprint": dataset_fingerprint,
                "dataset_size": dataset.size,
                "embedding": embedding_name,
            },
        )
    else:
        logger.info(
            "Mapa semantico: embeddings reutilizados desde cache con forma %s.",
            tuple(np.asarray(embeddings).shape),
        )

    embeddings = np.asarray(embeddings, dtype=np.float32)
    normalized_embeddings = _normalize_rows(embeddings)

    logger.info("Mapa semantico: ajustando PCA 2D sobre embeddings con forma %s.", tuple(embeddings.shape))
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    labels = np.asarray(dataset.labels, dtype=int)
    phishing_mask = labels == 1
    legitimate_mask = labels == 0

    centroid_phishing = embeddings[phishing_mask].mean(axis=0)
    centroid_legitimate = embeddings[legitimate_mask].mean(axis=0)
    centroid_coords = pca.transform(np.vstack([centroid_phishing, centroid_legitimate]))
    centroid_norms = _normalize_rows(np.vstack([centroid_phishing, centroid_legitimate]))

    _semantic_cache = {
        "cache_key": cache_key,
        "run_id": run_id,
        "embedding": embedding_name,
        "dataset_fingerprint": dataset_fingerprint,
        "texts": dataset.texts,
        "labels": labels,
        "coords": coords.astype(np.float32),
        "embeddings": embeddings,
        "normalized_embeddings": normalized_embeddings.astype(np.float32),
        "pca": pca,
        "centroids": {
            "phishing": centroid_phishing.astype(np.float32),
            "legitimate": centroid_legitimate.astype(np.float32),
        },
        "centroid_norms": {
            "phishing": centroid_norms[0].astype(np.float32),
            "legitimate": centroid_norms[1].astype(np.float32),
        },
        "centroid_coords": {
            "phishing": centroid_coords[0].astype(np.float32),
            "legitimate": centroid_coords[1].astype(np.float32),
        },
    }
    logger.info(
        "Mapa semantico: cache lista para run=%s embedding=%s con %s puntos.",
        run_id,
        embedding_name,
        len(dataset.texts),
    )
    return _semantic_cache


def _build_semantic_payload(subject: str, subject_embedding: np.ndarray, is_phishing: bool) -> dict | None:
    logger.info("Mapa semantico: iniciando payload para subject de longitud %s.", len(str(subject)))
    semantic = _prepare_semantic_cache()
    if not semantic:
        logger.warning("Mapa semantico: no se pudo obtener la cache semantica base.")
        return None

    emb = np.asarray(subject_embedding, dtype=np.float32).reshape(1, -1)
    emb_norm = _normalize_rows(emb)[0]
    point_2d = semantic["pca"].transform(emb)[0]

    similarities = semantic["normalized_embeddings"] @ emb_norm
    distances = 1.0 - similarities

    labels = semantic["labels"]
    phishing_indices = np.where(labels == 1)[0]
    legitimate_indices = np.where(labels == 0)[0]

    nearest_phishing_index = int(phishing_indices[np.argmin(distances[phishing_indices])])
    nearest_legitimate_index = int(legitimate_indices[np.argmin(distances[legitimate_indices])])

    centroid_distance_phishing = float(1.0 - np.dot(semantic["centroid_norms"]["phishing"], emb_norm))
    centroid_distance_legitimate = float(1.0 - np.dot(semantic["centroid_norms"]["legitimate"], emb_norm))

    sorted_indices = np.argsort(distances)
    max_neighbors = int(min(15, len(sorted_indices)))
    ranked_indices = [int(idx) for idx in sorted_indices[:max_neighbors]]
    neighbor_ranks = [-1] * len(distances)
    for rank, idx in enumerate(ranked_indices, start=1):
        neighbor_ranks[idx] = rank

    phishing_neighbors = sum(labels[idx] == 1 for idx in ranked_indices[:5])
    legitimate_neighbors = min(5, len(ranked_indices)) - phishing_neighbors
    dominant_label = "phishing" if phishing_neighbors >= legitimate_neighbors else "legitimate"

    interpretation = (
        "El asunto analizado cae en un vecindario semántico dominado por ejemplos de phishing."
        if dominant_label == "phishing"
        else "El asunto analizado cae en un vecindario semántico dominado por ejemplos legítimos."
    )

    points = []
    for idx, (coords, text, label) in enumerate(zip(semantic["coords"], semantic["texts"], labels)):
        points.append(
            {
                "id": idx,
                "x": round(float(coords[0]), 6),
                "y": round(float(coords[1]), 6),
                "subject_preview": _truncate_text(text, 180),
                "label": _label_name(int(label)),
                "similarity": round(float(similarities[idx]), 6),
                "distance": round(float(distances[idx]), 6),
                "neighbor_rank": neighbor_ranks[idx],
            }
        )

    payload = {
        "projection_method": "PCA",
        "distance_metric": "cosine",
        "note": "La posicion 2D proviene de PCA sobre embeddings reales. Las distancias y similitudes se calculan en el espacio semantico original del embedding.",
        "points": points,
        "analysis_point": {
            "x": round(float(point_2d[0]), 6),
            "y": round(float(point_2d[1]), 6),
            "subject_preview": _truncate_text(subject, 180),
            "predicted_label": "phishing" if is_phishing else "legitimate",
        },
        "centroids": {
            "phishing": {
                "x": round(float(semantic["centroid_coords"]["phishing"][0]), 6),
                "y": round(float(semantic["centroid_coords"]["phishing"][1]), 6),
            },
            "legitimate": {
                "x": round(float(semantic["centroid_coords"]["legitimate"][0]), 6),
                "y": round(float(semantic["centroid_coords"]["legitimate"][1]), 6),
            },
        },
        "nearest_by_class": {
            "phishing": {
                "index": nearest_phishing_index,
                "distance": round(float(distances[nearest_phishing_index]), 6),
                "similarity": round(float(similarities[nearest_phishing_index]), 6),
            },
            "legitimate": {
                "index": nearest_legitimate_index,
                "distance": round(float(distances[nearest_legitimate_index]), 6),
                "similarity": round(float(similarities[nearest_legitimate_index]), 6),
            },
        },
        "centroid_distances": {
            "phishing": round(centroid_distance_phishing, 6),
            "legitimate": round(centroid_distance_legitimate, 6),
        },
        "nearest_neighbor_indices": ranked_indices,
        "default_neighbor_count": min(5, max_neighbors),
        "max_neighbor_count": max_neighbors,
        "neighbor_summary": {
            "k": min(5, max_neighbors),
            "phishing": phishing_neighbors,
            "legitimate": legitimate_neighbors,
            "dominant_label": dominant_label,
        },
        "interpretation": interpretation,
    }
    logger.info(
        "Mapa semantico: payload construido con %s puntos, %s vecinos maximos y etiqueta dominante local '%s'.",
        len(points),
        max_neighbors,
        dominant_label,
    )
    return payload

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
        log_lines: list[str] = []

        def emit_log(content: str) -> str:
            log_lines.append(content)
            return json.dumps({"type": "log", "content": content}) + "\n"

        # --- CACHE CHECK ---
        run_id = _meta.get("run_id", "unknown")
        cached = get_cached_result(subject, run_id)
        if cached:
            semantic_map = None
            semantic_error = None
            prev_verdict = "PHISHING" if cached.get("status") == "phishing" else "LEGÍTIMO"
            prev_conf = cached.get("confidence", "?")
            yield json.dumps({"type": "log", "content": f"♻️ Caché forense activa — este asunto fue analizado previamente con el modelo '{run_id}'."}) + "\n"
            yield json.dumps({"type": "log", "content": f"   Veredicto almacenado: {prev_verdict} ({prev_conf}% de certeza). Recuperando todos los artefactos..."}) + "\n"
            try:
                yield json.dumps({"type": "log", "content": "🗺️ Reconstruyendo mapa semántico interactivo — proyectando el asunto en el espacio vectorial del dataset..."}) + "\n"
                emb = await asyncio.to_thread(_embedder.encode, [subject])
                semantic_map = await asyncio.to_thread(
                    _build_semantic_payload,
                    subject,
                    emb[0],
                    cached.get("status") == "phishing",
                )
                point_count = len(semantic_map.get("points", [])) if semantic_map else 0
                yield json.dumps({"type": "log", "content": f"✅ Mapa semántico listo — {point_count} puntos del dataset proyectados en 2D (PCA). Vecinos más cercanos calculados."}) + "\n"
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
                logger.warning("No se pudo reconstruir el mapa semantico desde cache: %s", exc)
                yield json.dumps({"type": "log", "content": f"⚠️ No se pudo reconstruir el mapa semántico para este resultado en caché: {semantic_error}"}) + "\n"
            yield json.dumps({"type": "log", "content": "📨 Restaurando resultado forense completo desde caché — saltando recálculo para máxima velocidad."}) + "\n"
            analysis_id, analysis_dir = _create_frontend_analysis_dir(subject)
            result_payload = _to_jsonable({
                **cached,
                "semantic_map": semantic_map,
                "semantic_error": semantic_error,
                "analysis_id": analysis_id,
                "artifact_dir": str(analysis_dir),
            })
            log_lines.extend([
                f"Resultado recuperado desde caché forense local (run={run_id}).",
                "Se reconstruyó el mapa semántico interactivo para esta recuperación." if semantic_map else "No fue posible reconstruir el mapa semántico interactivo durante la recuperación.",
                "Se entregó la respuesta final al frontend.",
            ])
            _persist_frontend_analysis_artifacts(
                analysis_id=analysis_id,
                analysis_dir=analysis_dir,
                result_data=result_payload,
                log_lines=log_lines,
                run_id=run_id,
            )
            yield json.dumps({"type": "result", "data": result_payload}) + "\n"
            return

        # 0. Check Model Status
        repo = _meta.get("embedding", "")
        status = get_model_status(repo)

        if status == "missing" or status == "incomplete":
            yield json.dumps({"type": "log", "content": f"📡 Modelo '{repo}' no encontrado en caché local de HuggingFace."}) + "\n"
            yield json.dumps({"type": "log", "content": "   Iniciando descarga desde HuggingFace Hub. Esto puede tardar varios minutos dependiendo del tamaño del modelo..."}) + "\n"
        else:
            yield json.dumps({"type": "log", "content": f"✅ Modelo '{repo}' verificado en caché local — no se requiere descarga."}) + "\n"

        # 1. Technical Details
        global _hw_msg
        model_name = repo or "Desconocido"
        classifier_name = _meta.get("classifier", "Desconocido")
        params_str = "Default"

        yield json.dumps({"type": "log", "content": f"💻 Hardware activo: {_hw_msg}"}) + "\n"
        if hasattr(_model, "get_params"):
            p = _model.get_params()
            relevant_keys = ["C", "kernel", "n_estimators", "max_depth", "alpha", "solver", "n_neighbors"]
            relevant = {k: v for k, v in p.items() if k in relevant_keys}
            if relevant:
                params_str = ", ".join([f"{k}={v}" for k, v in relevant.items()])

        yield json.dumps({"type": "log", "content": f"⚙️  Pipeline activo: embedder [{model_name}] + clasificador [{classifier_name}] (params: {params_str})"}) + "\n"
        yield json.dumps({"type": "log", "content": f"📝 Asunto recibido ({len(subject.split())} tokens): \"{subject[:80]}{'...' if len(subject) > 80 else ''}\""}) + "\n"

        # 2. Embedding & Prediction (NON-BLOCKING)
        yield json.dumps({"type": "log", "content": f"🔢 Vectorizando texto con [{model_name}] — convirtiendo el asunto en un vector numérico de alta dimensión que captura su significado semántico..."}) + "\n"

        # We use asyncio.to_thread to avoid blocking the main event loop
        emb = await asyncio.to_thread(_embedder.encode, [subject])
        yield json.dumps({"type": "log", "content": f"   Vector de embedding generado: dimensión {emb.shape[-1]}D. El texto queda representado como un punto en el espacio semántico."}) + "\n"

        yield json.dumps({"type": "log", "content": f"🎯 Clasificando con [{classifier_name}] — el modelo calcula la probabilidad de que este vector pertenezca a la clase phishing..."}) + "\n"
        probs = await asyncio.to_thread(_model.predict_proba, emb)
        probs = probs[0]

        phishing_probability = float(probs[1])
        is_phishing = bool(phishing_probability > 0.5)
        confidence = float(probs[1] if is_phishing else probs[0])
        prob_phishing_pct = phishing_probability * 100
        prob_legit_pct = (1.0 - phishing_probability) * 100
        yield json.dumps({"type": "log", "content": f"   Probabilidades brutas → Phishing: {prob_phishing_pct:.1f}%  |  Legítimo: {prob_legit_pct:.1f}%"}) + "\n"

        if is_phishing:
            yield json.dumps({"type": "log", "content": f"🚨 VEREDICTO PRELIMINAR: PHISHING ({prob_phishing_pct:.1f}% de probabilidad). Iniciando análisis de explicabilidad XAI..."}) + "\n"
        else:
            yield json.dumps({"type": "log", "content": f"🛡️  VEREDICTO PRELIMINAR: LEGÍTIMO ({prob_legit_pct:.1f}% de probabilidad). Ejecutando validación forense completa para confirmar..."}) + "\n"

        # 3. XAI Attribution
        n_tokens = len(subject.split())
        yield json.dumps({"type": "log", "content": f"🧠 XAI Leave-One-Out — eliminando cada palabra del asunto ({n_tokens} iteraciones) y midiendo cuánto cambia la probabilidad para identificar las más influyentes..."}) + "\n"
        lime_keywords = await asyncio.to_thread(compute_leave_one_out_keywords, subject, _embedder, _model)
        if lime_keywords:
            top = lime_keywords[0]
            direction = "↑ phishing" if top.get("positive") else "↓ phishing"
            yield json.dumps({"type": "log", "content": f"   Palabra más influyente: '{top.get('word', '?')}' (impacto: {top.get('impact', 0):.1f}% de atribución, dirección: {direction})"}) + "\n"
        yield json.dumps({"type": "log", "content": f"✅ Atribución completada — {len(lime_keywords)} palabras/frases clave identificadas como desencadenantes de la decisión."}) + "\n"

        # Counterfactual
        counterfactual = await asyncio.to_thread(_compute_counterfactual, subject, phishing_probability)
        if counterfactual:
            if counterfactual["flips_verdict"]:
                yield json.dumps({"type": "log", "content": f"🔀 Contrafactual: eliminar '{counterfactual['word']}' cambiaría P(phishing) de {counterfactual['original_prob']}% → {counterfactual['new_prob']}% → veredicto: {counterfactual['new_label'].upper()}"}) + "\n"
            else:
                yield json.dumps({"type": "log", "content": f"🔀 Contrafactual: la palabra más crítica es '{counterfactual['word']}' — eliminarla acercaría más al umbral ({counterfactual['original_prob']}% → {counterfactual['new_prob']}%) sin invertir el veredicto."}) + "\n"

        # Language warning
        lang_warning = detect_language_warning(subject)
        if lang_warning:
            yield json.dumps({"type": "log", "content": f"⚠️  Idioma: {lang_warning}"}) + "\n"

        # 4. Guardrail
        verdict = resolve_guardrailed_verdict(phishing_probability, lime_keywords)
        if verdict["decision_source"] == "attribution_guardrail" and not is_phishing:
            yield json.dumps({"type": "log", "content": "⚠️  GUARDRAIL FORENSE activado: el clasificador dio veredicto benigno en zona fronteriza, pero las palabras más influyentes son de alto riesgo. El sistema eleva el caso a PHISHING por coherencia XAI."}) + "\n"
        phishing_probability = float(verdict["probability"])
        is_phishing = bool(verdict["is_phishing"])
        confidence = float(phishing_probability if is_phishing else (1.0 - phishing_probability))
        final_label = "PHISHING" if is_phishing else "LEGÍTIMO"
        yield json.dumps({"type": "log", "content": f"   Veredicto final consolidado: {final_label} — certeza {confidence*100:.1f}%"}) + "\n"

        # 5. LLM Explanation
        yield json.dumps({"type": "log", "content": "📜 Generando razonamiento en lenguaje natural — el LLM explicará la decisión del modelo integrando las palabras clave y la probabilidad calculada..."}) + "\n"
        explanation = await asyncio.to_thread(llm_explainer.generate_explanation, subject, phishing_probability, lime_keywords)
        yield json.dumps({"type": "log", "content": "✅ Razonamiento generado y alineado con la atribución científica."}) + "\n"

        # 6. Synthetic Email Body
        yield json.dumps({"type": "log", "content": "✉️  Sintetizando cuerpo de correo de ejemplo — el LLM reconstruirá cómo podría verse un email real con este asunto para contextualizar la amenaza..."}) + "\n"
        fake_body = await asyncio.to_thread(llm_explainer.generate_email_body, subject, is_phishing)
        yield json.dumps({"type": "log", "content": "✅ Cuerpo sintético generado en el idioma detectado del asunto."}) + "\n"

        # 7. Semantic Map
        semantic_map = None
        semantic_error = None
        try:
            yield json.dumps({"type": "log", "content": "🗺️  Proyectando asunto en el mapa semántico — calculando su posición relativa respecto a los ejemplos del dataset en el espacio vectorial 2D (PCA)..."}) + "\n"
            semantic_map = await asyncio.to_thread(_build_semantic_payload, subject, emb[0], is_phishing)
            point_count = len(semantic_map.get("points", [])) if semantic_map else 0
            nb = semantic_map.get("neighbor_summary", {}) if semantic_map else {}
            dominant = nb.get("dominant_label", "?")
            ph_nb = nb.get("phishing", "?")
            lg_nb = nb.get("legitimate", "?")
            yield json.dumps({"type": "log", "content": f"✅ Mapa semántico listo — {point_count} puntos proyectados. Vecindario local (k=5): {ph_nb} phishing / {lg_nb} legítimos → zona dominante: {dominant.upper()}."}) + "\n"
        except FileNotFoundError:
            semantic_error = "FileNotFoundError: dataset fuente no disponible localmente"
            logger.warning("No se pudo generar el mapa semantico porque el dataset fuente no esta disponible localmente.")
            yield json.dumps({"type": "log", "content": "⚠️  El dataset fuente no está disponible en este entorno — el mapa semántico interactivo se omitirá."}) + "\n"
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            logger.warning("No se pudo construir el mapa semantico: %s", exc)
            yield json.dumps({"type": "log", "content": f"⚠️  Error construyendo el mapa semántico: {semantic_error}"}) + "\n"

        analysis_id, analysis_dir = _create_frontend_analysis_dir(subject)
        final_data = {
            "status": "phishing" if is_phishing else "safe",
            "is_phishing": is_phishing,
            "subject": subject,
            "confidence": round(confidence * 100, 1),
            "phishing_probability": round(phishing_probability * 100, 1),
            "keywords": lime_keywords,
            "counterfactual": counterfactual,
            "lang_warning": lang_warning,
            "explanation": explanation,
            "fake_body": fake_body,
            "semantic_map": semantic_map,
            "semantic_error": semantic_error,
            "analysis_id": analysis_id,
            "artifact_dir": str(analysis_dir),
        }
        log_lines.extend([
            f"Veredicto final: {final_data['status']} con certeza {final_data['confidence']}%.",
            f"Palabras clave detectadas: {len(lime_keywords)}.",
            "Razonamiento maestro generado por LLM.",
            "Cuerpo sintético del mensaje generado.",
            "Mapa semántico interactivo construido." if semantic_map else f"Mapa semántico omitido: {semantic_error}",
            "Respuesta final entregada al frontend.",
        ])
        _persist_frontend_analysis_artifacts(
            analysis_id=analysis_id,
            analysis_dir=analysis_dir,
            result_data=final_data,
            log_lines=log_lines,
            run_id=run_id,
        )

        # --- SAVE TO CACHE & HISTORY ---
        save_to_cache(final_data, run_id)
        history_index = save_to_history(final_data, embedding=emb[0])
        final_data["history_index"] = history_index
        yield json.dumps({"type": "log", "content": "💾 Resultado guardado en caché forense — próximas consultas de este asunto serán instantáneas."}) + "\n"
        yield json.dumps({"type": "result", "data": _to_jsonable(final_data)}) + "\n"

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


@app.post("/api/semantic-map")
async def get_semantic_map(req: SemanticRequest):
    global _embedder
    if not _embedder:
        raise HTTPException(status_code=500, detail="Embedder not loaded")

    subject = req.subject.strip()
    if not subject:
        raise HTTPException(status_code=400, detail="Subject is empty")

    try:
        emb = await asyncio.to_thread(_embedder.encode, [subject])
        semantic_map = await asyncio.to_thread(
            _build_semantic_payload,
            subject,
            emb[0],
            req.status == "phishing",
        )
        return {
            "semantic_map": _to_jsonable(semantic_map),
            "semantic_error": None,
        }
    except Exception as exc:
        semantic_error = f"{type(exc).__name__}: {exc}"
        logger.warning("No se pudo construir el mapa semantico on-demand: %s", exc)
        return {
            "semantic_map": None,
            "semantic_error": semantic_error,
        }


@app.post("/api/frontend-analysis-assets")
async def save_frontend_analysis_assets(req: FrontendAssetRequest):
    analysis_dir = FRONTEND_ANALYSES_DIR / req.analysis_id
    if not analysis_dir.exists():
        raise HTTPException(status_code=404, detail="Analysis artifact directory not found")

    saved = []

    if req.scatter_png_base64:
        scatter_payload = req.scatter_png_base64.split(",", 1)[-1]
        (analysis_dir / "semantic_scatter.png").write_bytes(base64.b64decode(scatter_payload))
        saved.append("semantic_scatter.png")

    if req.pdf_base64:
        pdf_payload = req.pdf_base64.split(",", 1)[-1]
        (analysis_dir / "forensic_report.pdf").write_bytes(base64.b64decode(pdf_payload))
        saved.append("forensic_report.pdf")

    return {"saved": saved}

# Serve Frontend — no-cache so browsers always load the latest version
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse

class NoCacheStaticMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/") and not request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

app.add_middleware(NoCacheStaticMiddleware)
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
