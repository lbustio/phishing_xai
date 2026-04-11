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
import torch.nn as nn
import transformers
import sentence_transformers
import asyncio
import csv
import requests
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

from config.paths import RUNS_DIR, TABLES_DIR, HF_CACHE_HINT_DIR, RESULTS_DIR, TEMP_DEMO_XAI_DIR, DATA_DIR, FRONTEND_ANALYSES_DIR
from config.experiment import get_embedding_config, XAI_CONFIG, PRIMARY_METRIC
from src.data_loader import load_dataset
from src.embedding_store import EmbeddingStore
from src.embeddings.cache_utils import get_hf_model_cache_status
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
    return get_hf_model_cache_status(repo_name, HF_CACHE_HINT_DIR)

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
    persisted = {k: v for k, v in result_data.items() if k != "semantic_map"}
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **persisted
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
        except:
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
            try:
                yield json.dumps({"type": "log", "content": "Preparando mapa semántico interactivo con embeddings reales del dataset..."}) + "\n"
                emb = await asyncio.to_thread(_embedder.encode, [subject])
                semantic_map = await asyncio.to_thread(
                    _build_semantic_payload,
                    subject,
                    emb[0],
                    cached.get("status") == "phishing",
                )
                point_count = len(semantic_map.get("points", [])) if semantic_map else 0
                yield json.dumps({"type": "log", "content": f"✅ Mapa semántico preparado correctamente ({point_count} puntos reales)."}) + "\n"
                yield json.dumps({"type": "log", "content": "📦 Serializando payload semántico para enviarlo al navegador..."}) + "\n"
            except Exception as exc:
                semantic_error = f"{type(exc).__name__}: {exc}"
                logger.warning("No se pudo reconstruir el mapa semantico desde cache: %s", exc)
                yield json.dumps({"type": "log", "content": f"⚠️ No fue posible regenerar el mapa semántico interactivo para este resultado en caché. Detalle: {semantic_error}"}) + "\n"
            yield json.dumps({"type": "log", "content": "♻️ Resultados recuperados de la caché forense local."}) + "\n"
            yield json.dumps({"type": "log", "content": "✅ Análisis previo detectado: Restaurando interpretatibilidad..."}) + "\n"
            analysis_id, analysis_dir = _create_frontend_analysis_dir(subject)
            result_payload = _to_jsonable({
                **cached,
                "semantic_map": semantic_map,
                "semantic_error": semantic_error,
                "analysis_id": analysis_id,
                "artifact_dir": str(analysis_dir),
            })
            log_lines.extend([
                "Resultados recuperados desde caché local del PowerToy.",
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
            yield json.dumps({"type": "log", "content": "📨 Entregando respuesta final al frontend..."}) + "\n"
            yield json.dumps({"type": "result", "data": result_payload}) + "\n"
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
        semantic_map = None
        semantic_error = None
        try:
            yield json.dumps({"type": "log", "content": "🗺️ Construyendo visualización semántica real del dataset para la vista interactiva..."}) + "\n"
            semantic_map = await asyncio.to_thread(_build_semantic_payload, subject, emb[0], is_phishing)
            point_count = len(semantic_map.get("points", [])) if semantic_map else 0
            yield json.dumps({"type": "log", "content": f"✅ Mapa semántico preparado correctamente ({point_count} puntos reales)."}) + "\n"
            yield json.dumps({"type": "log", "content": "📦 Serializando payload semántico para enviarlo al navegador..."}) + "\n"
        except FileNotFoundError:
            semantic_error = "FileNotFoundError: dataset fuente no disponible localmente"
            logger.warning("No se pudo generar el mapa semantico porque el dataset fuente no esta disponible localmente.")
            yield json.dumps({"type": "log", "content": f"⚠️ El dataset fuente no está disponible localmente; el mapa semántico interactivo se omitirá en esta ejecución. Detalle: {semantic_error}"}) + "\n"
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            logger.warning("No se pudo construir el mapa semantico: %s", exc)
            yield json.dumps({"type": "log", "content": f"⚠️ No fue posible construir el mapa semántico interactivo con los datos reales de esta configuración. Detalle: {semantic_error}"}) + "\n"

        analysis_id, analysis_dir = _create_frontend_analysis_dir(subject)
        final_data = {
            "status": "phishing" if is_phishing else "safe",
            "subject": subject,
            "confidence": round(confidence * 100, 1),
            "keywords": lime_keywords,
            "explanation": explanation,
            "fake_body": fake_body,
            "semantic_map": semantic_map,
            "semantic_error": semantic_error,
            "analysis_id": analysis_id,
            "artifact_dir": str(analysis_dir),
        }
        log_lines.extend([
            f"Veredicto final: {final_data['status']} con confianza {final_data['confidence']}%.",
            f"Palabras relevantes detectadas: {len(lime_keywords)}.",
            "Se generó el razonamiento maestro.",
            "Se generó el cuerpo sintético del mensaje.",
            "Se construyó el mapa semántico interactivo." if semantic_map else f"No se pudo construir el mapa semántico interactivo. Detalle: {semantic_error}",
            "Se entregó la respuesta final al frontend.",
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
        save_to_history(final_data)
        
        yield json.dumps({"type": "log", "content": "📨 Entregando respuesta final al frontend..."}) + "\n"
        yield json.dumps({"type": "result", "data": _to_jsonable(final_data)}) + "\n"
        yield json.dumps({"type": "log", "content": "💾 Análisis finalizado y guardado exitosamente en la caché forense."}) + "\n"

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

# Serve Frontend
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
