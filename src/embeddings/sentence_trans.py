"""
src/embeddings/sentence_trans.py
=================================
Embedding backend for SentenceTransformer-compatible models.

This covers the majority of models in the evaluation suite:
  - all-mpnet-base-v2 (MPNet)
  - hkunlp/instructor-large (INSTRUCTOR)
  - BAAI/bge-m3, BAAI/bge-large-en-v1.5
  - intfloat/e5-large-v2
  - thenlper/gte-large
  - jinaai/jina-embeddings-v3

For instruction-tuned models (INSTRUCTOR, E5), the 'query_prefix' acts as the
task-specific instruction prepended to every input.

Memory management
-----------------
The model is released from memory (including GPU cache) after encode() returns,
because the pipeline processes one embedding model at a time and holding all
models in memory simultaneously would be unfeasible.
"""

import gc
import logging
from typing import List, Optional

import numpy as np
import torch

from .base import BaseEmbedder

logger = logging.getLogger("phishing_xai.embeddings.sentence_trans")


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Wraps any SentenceTransformer-compatible model.

    Parameters
    ----------
    name               : str          – Friendly name for logging.
    repo               : str          – HuggingFace model id.
    query_prefix       : str          – Task instruction/prefix.
    batch_size         : int          – Samples per forward pass.
    device             : torch.device – Target compute device.
    trust_remote_code  : bool         – Required by some models (e.g. Jina, Qwen).
    """

    def __init__(
        self,
        name: str,
        repo: str,
        query_prefix: str,
        instruction_mode: str,
        batch_size: int,
        device: torch.device,
        precision: torch.dtype = torch.float32,
        trust_remote_code: bool = False,
        cache_folder: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(name, repo, query_prefix, batch_size)
        self.device            = device
        self.precision         = precision
        self.instruction_mode  = instruction_mode
        self.trust_remote_code = trust_remote_code
        self.cache_folder      = cache_folder
        self.local_files_only  = local_files_only
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "The 'sentence-transformers' package is required. "
                    "Install it with: pip install sentence-transformers"
                )

            logger.info(
                f"Loading SentenceTransformer model '{self.repo}' from cache={self.cache_folder} "
                f"onto {str(self.device).upper()} with precision {self.precision} "
                f"(trust_remote_code={self.trust_remote_code})..."
            )
            self._model = SentenceTransformer(
                self.repo,
                device=str(self.device),
                trust_remote_code=self.trust_remote_code,
                cache_folder=self.cache_folder,
                local_files_only=self.local_files_only,
                model_kwargs={"torch_dtype": self.precision}
            )
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Compute dense embeddings for a list of texts.

        The SentenceTransformer library handles tokenisation, batching, and
        GPU/CPU placement internally.  We only add prefix injection and explicit
        memory cleanup after encoding is complete.

        Parameters
        ----------
        texts : List[str]  – Raw input texts (prefix not yet applied).

        Returns
        -------
        np.ndarray  – Float32 matrix of shape (len(texts), dim).
        """
        model = self._ensure_model()

        if self.instruction_mode == "pair" and self.query_prefix:
            texts_to_encode = [[self.query_prefix, text] for text in texts]
        else:
            texts_to_encode = self._apply_prefix(texts)

        n_total   = len(texts_to_encode)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Encoding {n_total} email subjects in {n_batches} batches "
            f"(batch_size={self.batch_size})..."
        )

        all_embs: List[np.ndarray] = []
        for batch_idx, start in enumerate(range(0, n_total, self.batch_size), 1):
            end   = min(start + self.batch_size, n_total)
            batch = texts_to_encode[start:end]

            # Log progress every 10 batches (or always for small datasets)
            if n_batches <= 20 or batch_idx % 10 == 0 or batch_idx == n_batches:
                logger.info(
                    f"  [{self.name}] Batch {batch_idx}/{n_batches} "
                    f"({start}–{end} of {n_total} subjects)..."
                )

            emb = model.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.batch_size,
            )
            all_embs.append(emb.astype(np.float32))

        result = np.vstack(all_embs)
        logger.info(
            f"Encoding complete for '{self.name}'. "
            f"Embedding matrix shape: {result.shape} "
            f"(samples × dimensions)."
        )

        # ── Explicit memory cleanup ────────────────────────────────────────────
        return result

    def release_resources(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
