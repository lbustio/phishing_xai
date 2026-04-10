from __future__ import annotations
import logging
from pathlib import Path

import joblib
import numpy as np

from config.experiment import get_embedding_config
from src.embeddings.factory import get_embedder
from src.xai.word_ablation_explainer import LIMEExplainer
from src.xai.shap_explainer import SHAPExplainer
from src.xai.llm_explainer_v2 import NaturalLanguageExplainer

logger = logging.getLogger("phishing_xai.xai_runner")

def _select_representative_indices(labels: list[int], predictions: np.ndarray, n_examples: int) -> list[int]:
    labels_array = np.asarray(labels)
    n_total = len(labels_array)
    rng = np.random.default_rng(42)
    # Selecciona índices aleatorios sin reemplazo (o la lógica que tuvieras aquí originalmente)
    return rng.choice(n_total, min(n_examples, n_total), replace=False).tolist()

def run_xai(
    *,
    embedding_id: str,
    classifier_name: str,
    model_path: Path,
    texts: list[str],
    labels: list[int],
    device,
    xai_lime_dir: Path,
    xai_shap_dir: Path,
    n_examples: int,
) -> None:
    logger.info(
        "Iniciando fase XAI para la mejor combinacion detectada: [%s] + [%s].",
        embedding_id,
        classifier_name,
    )
    
    embedding_config = get_embedding_config(embedding_id)
    embedder = get_embedder(
        name=embedding_id,
        model_config=embedding_config,
        device=device,
        checkpoint_dir=xai_shap_dir.parent,
    )

    # --- LA ÚNICA MODIFICACIÓN: CARGA SEGURA PARA WINDOWS ---
    ruta_absoluta = Path(model_path).resolve()
    model = joblib.load(str(ruta_absoluta))
    # --------------------------------------------------------

    # --- TU CÓDIGO ORIGINAL INTACTO ---
    try:
        def predict_proba_pipeline(batch_texts) -> np.ndarray:
            # 1. Usar len() es seguro tanto para listas de Python como para Numpy Arrays
            if len(batch_texts) == 0:
                return np.empty((0, 2), dtype=np.float32)
            
            # 2. Si SHAP envía un Numpy Array, lo forzamos a lista normal de strings
            if isinstance(batch_texts, np.ndarray):
                batch_texts = batch_texts.tolist()
                
            batch_embeddings = embedder.encode(batch_texts)
            return model.predict_proba(batch_embeddings)

        logger.info("Calculando predicciones del modelo final sobre el dataset completo para seleccionar casos representativos.")
        full_embeddings = embedder.encode(texts)
        predictions = model.predict(full_embeddings)
        
        selected_indices = _select_representative_indices(labels, predictions, n_examples)
        selected_texts = [texts[index] for index in selected_indices]
        selected_labels = [labels[index] for index in selected_indices]

        logger.info(
            "Se explicaran %s instancias representativas con LIME y SHAP.",
            len(selected_indices),
        )

        lime = LIMEExplainer(output_dir=xai_lime_dir)
        lime_results = lime.explain_batch(selected_texts, predict_proba_pipeline, selected_labels)

        shap = SHAPExplainer(output_dir=xai_shap_dir)
        shap_results = shap.explain_batch(selected_texts, predict_proba_pipeline, selected_labels)

        logger.info("[XAI] Procesando resultados algorítmicos para generar explicaciones en lenguaje natural...")
        lime_keywords_batch = []
        for r in lime_results:
            kw = [str(w) for w, weight in r.get("features", []) if float(weight) > 0]
            lime_keywords_batch.append(kw)

        shap_keywords_batch = []
        for r in shap_results:
            kw = []
            if "words" in r and "shap_values" in r:
                for word, val in zip(r["words"], r["shap_values"]):
                    if float(val) > 0:
                        kw.append(str(word))
            shap_keywords_batch.append(kw)

        llm_explainer = NaturalLanguageExplainer()
        llm_explainer.process_batch(
            subjects=selected_texts,
            lime_keywords_batch=lime_keywords_batch,
            shap_keywords_batch=shap_keywords_batch,
            output_file=xai_lime_dir.parent / "llm_explanations.json"
        )
    finally:
        embedder.release_resources()
        
    logger.info("[XAI] Explanation phase completed successfully.")
