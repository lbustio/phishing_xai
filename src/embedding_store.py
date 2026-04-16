from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from config.paths import EMBEDDING_CACHE_DIR
from src.utils.io_utils import atomic_write_json, read_json

logger = logging.getLogger("phishing_xai.embedding_store")


def _safe_slug(text: str) -> str:
    return text.replace("/", "__").replace("\\", "__").replace(":", "_").replace(" ", "_")


class EmbeddingStore:
    def __init__(self, dataset_fingerprint: str) -> None:
        self.dataset_fingerprint = dataset_fingerprint
        self.dataset_dir = EMBEDDING_CACHE_DIR / dataset_fingerprint
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

    def get_model_dir(self, embedding_id: str) -> Path:
        model_dir = self.dataset_dir / _safe_slug(embedding_id)
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    def get_embedding_file(self, embedding_id: str) -> Path:
        return self.get_model_dir(embedding_id) / "embeddings.npy"

    def get_metadata_file(self, embedding_id: str) -> Path:
        return self.get_model_dir(embedding_id) / "metadata.json"

    def load(self, embedding_id: str) -> np.ndarray | None:
        embedding_file = self.get_embedding_file(embedding_id)
        if not embedding_file.exists():
            return None
        logger.info(
            "Se reutilizara la cache persistente de embeddings para '%s' desde '%s'.",
            embedding_id,
            embedding_file,
        )
        return np.load(embedding_file)

    def save(self, embedding_id: str, embeddings: np.ndarray, metadata: dict) -> None:
        embedding_file = self.get_embedding_file(embedding_id)
        np.save(embedding_file, embeddings)
        atomic_write_json(self.get_metadata_file(embedding_id), metadata)
        logger.info(
            "Embeddings persistidos para '%s' en '%s' con forma %s.",
            embedding_id,
            embedding_file,
            embeddings.shape,
        )

    def load_metadata(self, embedding_id: str) -> dict:
        return read_json(self.get_metadata_file(embedding_id), {})
