"""
src/embeddings/hf_encoder.py
=============================
Embedding backend for standard HuggingFace Transformer encoder models that are
NOT packaged as SentenceTransformer modules.

Primary use case: distilbert-base-uncased and similar masked language models
whose HuggingFace checkpoints do not include a sentence-level pooling layer.

Approach
--------
  1. Load AutoTokenizer and AutoModel from HuggingFace Hub.
  2. For each batch of texts:
     a. Tokenise with padding and truncation.
     b. Run a forward pass (no gradient).
     c. Apply attention-mask-weighted mean pooling over the token dimension
        to obtain a single fixed-length vector per sequence.
  3. Stack all batch outputs into a single float32 matrix.

Why mean pooling over the [CLS] token?
  The [CLS] token in models NOT fine-tuned for sentence similarity often
  encodes next-sentence prediction information rather than sentence-level
  semantics.  Mean pooling over non-padding tokens consistently outperforms
  [CLS]-only extraction on text classification tasks with general encoders.
"""

import gc
import logging
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor

from .base import BaseEmbedder

logger = logging.getLogger("phishing_xai.embeddings.hf_encoder")


def _mean_pool(last_hidden_state: Tensor, attention_mask: Tensor) -> Tensor:
    """
    Compute attention-mask-weighted mean pooling over the sequence dimension.

    Parameters
    ----------
    last_hidden_state : Tensor  – Shape (B, L, D): last transformer hidden states.
    attention_mask    : Tensor  – Shape (B, L):    1 for real tokens, 0 for padding.

    Returns
    -------
    Tensor  – Shape (B, D): one embedding vector per sequence.
    """
    # Expand mask to match hidden state dimensions:  (B, L) → (B, L, D)
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    # Zero out padding positions, then sum over sequence axis
    sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
    # Divide by the number of non-padding tokens (clamped to ≥ 1e-9 to avoid /0)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask


class HFEncoderEmbedder(BaseEmbedder):
    """
    Generic HuggingFace Transformer encoder with mean-pooling extraction.

    Parameters
    ----------
    name         : str          – Friendly model name (for logging).
    repo         : str          – HuggingFace model identifier.
    query_prefix : str          – Text prepended to inputs (usually empty for encoders).
    batch_size   : int          – Forward-pass batch size.
    device       : torch.device – Target compute device.
    max_length   : int          – Maximum tokenisation length (default: 512).
    """

    def __init__(
        self,
        name: str,
        repo: str,
        query_prefix: str,
        batch_size: int,
        device: torch.device,
        precision: torch.dtype = torch.float32,
        max_length: int = 512,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__(name, repo, query_prefix, batch_size)
        self.device     = device
        self.precision  = precision
        self.max_length = max_length
        self.cache_dir  = cache_dir
        self._tokenizer = None
        self._model = None

    def _ensure_model_and_tokenizer(self):
        if self._model is None or self._tokenizer is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError:
                raise ImportError(
                    "The 'transformers' package is required. "
                    "Install it with: pip install transformers"
                )

            logger.info(
                f"Loading HuggingFace encoder model '{self.repo}' from cache={self.cache_dir} "
                f"onto {str(self.device).upper()} with precision {self.precision}..."
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self.repo, cache_dir=self.cache_dir)
            self._model = AutoModel.from_pretrained(
                self.repo, 
                cache_dir=self.cache_dir,
                torch_dtype=self.precision,
                low_cpu_mem_usage=True
            )
            
            # Map to device. If torch_dtype is provided, to() is better after loading
            # but for LLMs we use device_map. For small encoders, to(device) is fine.
            self._model.to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Tokenise and encode a list of texts via mean-pooled Transformer hidden states.

        Parameters
        ----------
        texts : List[str]  – Raw input texts (prefix not yet applied).

        Returns
        -------
        np.ndarray  – Float32 array of shape (len(texts), hidden_dim).
        """
        tokenizer, model = self._ensure_model_and_tokenizer()

        texts_with_prefix = self._apply_prefix(texts)
        n_total   = len(texts_with_prefix)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Encoding {n_total} email subjects using '{self.name}' "
            f"({n_batches} batches of up to {self.batch_size})..."
        )

        all_embs: List[np.ndarray] = []

        with torch.no_grad():
            for batch_idx, start in enumerate(range(0, n_total, self.batch_size), 1):
                end   = min(start + self.batch_size, n_total)
                batch = texts_with_prefix[start:end]

                if n_batches <= 20 or batch_idx % 10 == 0 or batch_idx == n_batches:
                    logger.info(
                        f"  [{self.name}] Batch {batch_idx}/{n_batches} "
                        f"(subjects {start}–{end})..."
                    )

                # Tokenise: padding to the longest sequence in the batch,
                # truncation to max_length.
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}

                outputs = model(**enc)

                # Extract and pool: (B, L, D) → (B, D)
                pooled = _mean_pool(
                    outputs.last_hidden_state,
                    enc["attention_mask"],
                )

                all_embs.append(pooled.cpu().float().numpy())

        result = np.vstack(all_embs)

        logger.info(
            f"Encoding complete for '{self.name}'. "
            f"Embedding matrix shape: {result.shape}."
        )

        # ── Explicit memory cleanup ────────────────────────────────────────────
        return result.astype(np.float32)

    def release_resources(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
