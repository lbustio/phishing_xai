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

from abc import ABC, abstractmethod
from typing import List

import numpy as np


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
