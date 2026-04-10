"""
src/embeddings/factory.py
=========================
Factory function that instantiates the correct embedding backend given a model
configuration dict (as defined in config/experiment.py).

The factory is the single place where the 'type' field in each embedding config
entry is translated into a concrete BaseEmbedder subclass.  No other module
needs to know which backend class to use.

Supported types (config field: 'type')
---------------------------------------
  'sentence_transformer' → SentenceTransformerEmbedder
      Used for: MPNet, INSTRUCTOR, BGE-M3, E5-Large, BGE-Large, GTE-Large, Jina-v3
  'hf_encoder'           → HFEncoderEmbedder (mean pooling over AutoModel)
      Used for: DistilBERT
  'large_llm'            → LLMEmbedder (last-token pooling, optional FP16)
      Used for: E5-Mistral, SFR-Mistral, SFR-2, GTE-Qwen2, Llama-3.1

Usage
-----
    from src.embeddings.factory import get_embedder
    embedder = get_embedder("MPNet", EMBEDDING_MODELS["MPNet"], device, ckpt_dir)
    X = embedder.encode(texts)
"""

import logging
from pathlib import Path
from typing import Optional

import torch

from config.paths import HF_CACHE_HINT_DIR
from .base import BaseEmbedder
from .sentence_trans import SentenceTransformerEmbedder
from .hf_encoder import HFEncoderEmbedder
from .llm_embedder import LLMEmbedder

logger = logging.getLogger("phishing_xai.embeddings.factory")

# Models known to require last-token pooling rather than mean pooling.
# All decoder-based LLMs fall in this category.
_LAST_TOKEN_POOL_MODELS = {
    "intfloat/e5-mistral-7b-instruct",
    "Salesforce/SFR-Embedding-Mistral",
    "Salesforce/SFR-Embedding-2_R",
    "Alibaba-NLP/gte-Qwen2-7B-instruct",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def get_embedder(
    name: str,
    model_config: dict,
    device: torch.device,
    precision: torch.dtype = torch.float32,
    checkpoint_dir: Optional[Path] = None,
) -> BaseEmbedder:
    """
    Instantiate and return the appropriate embedding backend.

    Parameters
    ----------
    name           : str           – Friendly model name (key in EMBEDDING_MODELS).
    model_config   : dict          – Entry from EMBEDDING_MODELS in config/experiment.py.
    device         : torch.device  – Compute device (cuda / mps / cpu).
    precision      : torch.dtype   – Target precision (default fp32).
    checkpoint_dir : Path, optional – Directory for LLM partial-progress checkpoints.

    Returns
    -------
    BaseEmbedder  – Configured but not yet executed; call .encode(texts) to run.

    Raises
    ------
    ValueError  – If 'type' field is unrecognised.
    """
    model_type  = model_config["type"]
    repo        = model_config["repo"]
    prefix      = model_config.get("query_prefix", "")
    instruction_mode = model_config.get("instruction_mode", "prefix")
    batch_size  = model_config.get("batch_size", 32)
    trust_rc    = model_config.get("trust_remote_code", False)

    logger.info(
        f"Creating embedder for '{name}' "
        f"(type={model_type}, repo={repo}, device={device}, "
        f"batch_size={batch_size}, prefix={repr(prefix)[:60]})."
    )

    if model_type == "sentence_transformer":
        return SentenceTransformerEmbedder(
            name=name,
            repo=repo,
            query_prefix=prefix,
            instruction_mode=instruction_mode,
            batch_size=batch_size,
            device=device,
            precision=precision,
            trust_remote_code=trust_rc,
            cache_folder=str(HF_CACHE_HINT_DIR),
        )

    elif model_type == "hf_encoder":
        return HFEncoderEmbedder(
            name=name,
            repo=repo,
            query_prefix=prefix,
            batch_size=batch_size,
            device=device,
            precision=precision,
            max_length=512,
            cache_dir=str(HF_CACHE_HINT_DIR),
        )

    elif model_type == "large_llm":
        use_ltp = repo in _LAST_TOKEN_POOL_MODELS
        if not use_ltp:
            logger.debug(
                f"'{repo}' not in last-token-pool whitelist; using mean pooling."
            )
        return LLMEmbedder(
            name=name,
            repo=repo,
            query_prefix=prefix,
            batch_size=batch_size,
            device=device,
            precision=precision,
            trust_remote_code=trust_rc,
            checkpoint_dir=checkpoint_dir,
            use_last_token_pool=use_ltp,
            cache_dir=str(HF_CACHE_HINT_DIR),
        )

    else:
        raise ValueError(
            f"Unknown embedding type '{model_type}' for model '{name}'. "
            "Supported types: 'sentence_transformer', 'hf_encoder', 'large_llm'."
        )
