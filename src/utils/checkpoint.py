from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config.experiment import PRIMARY_METRIC
from src.utils.io_utils import atomic_write_json, read_json

logger = logging.getLogger("phishing_xai.checkpoint")


class ExperimentCheckpoint:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = read_json(
            path,
            {
                "completed": {},
                "skipped_embeddings": {},
                "best_result": None,
            },
        )

    @staticmethod
    def make_key(embedding_id: str, classifier_name: str) -> str:
        return f"{embedding_id}::{classifier_name}"

    def save(self) -> None:
        atomic_write_json(self.path, self.state)

    def is_done(self, embedding_id: str, classifier_name: str) -> bool:
        return self.make_key(embedding_id, classifier_name) in self.state["completed"]

    def get_result(self, embedding_id: str, classifier_name: str) -> dict[str, Any] | None:
        return self.state["completed"].get(self.make_key(embedding_id, classifier_name))

    def record_result(self, embedding_id: str, classifier_name: str, payload: dict[str, Any]) -> None:
        key = self.make_key(embedding_id, classifier_name)
        self.state["completed"][key] = payload

        current_best = self.state.get("best_result")
        current_score = -1.0 if current_best is None else current_best["metrics"].get(PRIMARY_METRIC, -1.0)
        new_score = payload["metrics"].get(PRIMARY_METRIC, -1.0)
        if new_score > current_score:
            self.state["best_result"] = {
                "embedding": embedding_id,
                "classifier": classifier_name,
                **payload,
            }
            logger.info(
                "Nuevo mejor resultado global registrado: [%s] + [%s] con %s=%.4f.",
                embedding_id,
                classifier_name,
                PRIMARY_METRIC,
                new_score,
            )
        self.save()

    def mark_embedding_skipped(self, embedding_id: str, reason: str) -> None:
        self.state["skipped_embeddings"][embedding_id] = reason
        self.save()

    def is_embedding_skipped(self, embedding_id: str) -> str | None:
        return self.state["skipped_embeddings"].get(embedding_id)

    def get_best_result(self) -> dict[str, Any] | None:
        return self.state.get("best_result")

    def summary(self) -> str:
        completed = len(self.state["completed"])
        skipped = len(self.state["skipped_embeddings"])
        best = self.get_best_result()
        if best is None:
            return f"Checkpoint activo: {completed} combinaciones completadas, {skipped} embeddings omitidos."
        return (
            "Checkpoint activo: "
            f"{completed} combinaciones completadas, {skipped} embeddings omitidos. "
            f"Mejor hasta ahora: [{best['embedding']}] + [{best['classifier']}] "
            f"{PRIMARY_METRIC}={best['metrics'].get(PRIMARY_METRIC, 0.0):.4f}."
        )
