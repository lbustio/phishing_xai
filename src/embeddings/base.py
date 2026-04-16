"""
src/embeddings/base.py
======================
Abstract base class for all embedding backends.

All concrete embedders (SentenceTransformerEmbedder, HFEncoderEmbedder,
LLMEmbedder) inherit from BaseEmbedder and must implement:

  encode(texts: List[str]) -> np.ndarray
    Returns a float32 matrix of shape (n_texts, embedding_dim).

The encode() method is the single integration point that the rest of the
pipeline uses: the trainer, the checkpoint system, and the XAI wrappers all
call embed.encode(texts) without knowing which backend is in use.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List

import numpy as np


@contextmanager
def hf_offline_if_cached(local_files_only: bool):
    """Set HF_HUB_OFFLINE=1 during model loading when the model is already cached.

    local_files_only=True prevents weight downloads but huggingface_hub still
    fires HTTP requests to check metadata, commits and discussions. This context
    manager cuts all network traffic by setting HF_HUB_OFFLINE, which is the
    only reliable way to enforce fully offline behaviour.
    """
    if not local_files_only:
        yield
        return
    prev_hub = os.environ.get("HF_HUB_OFFLINE")
    prev_tf  = os.environ.get("TRANSFORMERS_OFFLINE")
    os.environ["HF_HUB_OFFLINE"]      = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        # Restore exactly what was there before (including "not set")
        if prev_hub is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev_hub
        if prev_tf is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = prev_tf


class BaseEmbedder(ABC):
    """
    Abstract base class defining the interface for all embedding backends.

    Parameters
    ----------
    name         : str   – Friendly name used in logs and results tables.
    repo         : str   – HuggingFace model identifier (for logging/caching).
    query_prefix : str   – String prepended to every input text at encode time.
                           Empty string means no prefix is applied.
    batch_size   : int   – Number of texts processed per forward pass.
    """

    def __init__(
        self,
        name: str,
        repo: str,
        query_prefix: str,
        batch_size: int,
    ) -> None:
        self.name         = name
        self.repo         = repo
        self.query_prefix = query_prefix
        self.batch_size   = batch_size

    def _apply_prefix(self, texts: List[str]) -> List[str]:
        """
        Prepend query_prefix to each text if a prefix is configured.

        This is called internally by concrete encode() implementations.
        """
        if self.query_prefix:
            return [self.query_prefix + t for t in texts]
        return texts

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into a dense embedding matrix.

        Parameters
        ----------
        texts : List[str]  – Raw (unprefixed) input texts.

        Returns
        -------
        np.ndarray  – Shape (len(texts), embedding_dim), dtype float32.
        """
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"repo={self.repo!r}, "
            f"prefix={self.query_prefix!r}, "
            f"batch_size={self.batch_size})"
        )

    def release_resources(self) -> None:
        """Optional hook for embedders that keep models/tokenizers alive in memory."""
        return None
