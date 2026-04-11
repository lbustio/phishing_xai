import gc
import hashlib
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .base import BaseEmbedder

logger = logging.getLogger("phishing_xai.embeddings.llm_embedder")

try:
    from transformers.cache_utils import DynamicCache

    if hasattr(DynamicCache, "get_seq_length") and not hasattr(DynamicCache, "get_usable_length"):

        def _get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
            return self.get_seq_length(layer_idx)

        DynamicCache.get_usable_length = _get_usable_length  # type: ignore[attr-defined]
except Exception:
    pass


def _last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    seq_lengths = attention_mask.sum(dim=1) - 1
    seq_lengths = seq_lengths.clamp(min=0)
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        seq_lengths,
    ]


def _mean_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    summed = (last_hidden_states * mask_expanded).sum(dim=1)
    denom = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class LLMEmbedder(BaseEmbedder):
    def __init__(
        self,
        name: str,
        repo: str,
        query_prefix: str,
        batch_size: int,
        device: torch.device,
        trust_remote_code: bool = False,
        checkpoint_dir: Optional[Path] = None,
        use_last_token_pool: bool = True,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__(name, repo, query_prefix, batch_size)
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.checkpoint_dir = checkpoint_dir
        self.use_last_token_pool = use_last_token_pool
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None

    def _ckpt_path(self, texts: List[str]) -> Optional[Path]:
        if self.checkpoint_dir is None:
            return None
        sample = "\n".join(texts[:100])
        digest = hashlib.md5(sample.encode()).hexdigest()[:8]
        return self.checkpoint_dir / f"partial_{self.name.replace('/', '__')}_{digest}.pkl"

    def _load_partial(self, texts: List[str]) -> tuple[list[np.ndarray], int]:
        path = self._ckpt_path(texts)
        if path and path.exists():
            try:
                with open(path, "rb") as handle:
                    payload = pickle.load(handle)
                embeddings = payload.get("embeddings", [])
                start_idx = sum(chunk.shape[0] for chunk in embeddings)
                logger.info(
                    "Se encontro un checkpoint parcial de embeddings para '%s'. Se retomara desde el indice %s.",
                    self.name,
                    start_idx,
                )
                return embeddings, start_idx
            except Exception as exc:
                logger.warning(
                    "El checkpoint parcial de '%s' no pudo reutilizarse (%s). Se recalculara desde cero.",
                    self.name,
                    exc,
                )
        return [], 0

    def _save_partial(self, embeddings: list[np.ndarray], texts: List[str]) -> None:
        path = self._ckpt_path(texts)
        if path is None:
            return
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as handle:
            pickle.dump({"repo": self.repo, "embeddings": embeddings}, handle)
        tmp.replace(path)

    def _delete_partial(self, texts: List[str]) -> None:
        path = self._ckpt_path(texts)
        if path and path.exists():
            path.unlink(missing_ok=True)

    def _ensure_model_and_tokenizer(self):
        if self._model is not None and self._tokenizer is not None:
            return self._tokenizer, self._model

        from transformers import AutoModel, AutoTokenizer

        # Use the precision suggested by the factory/hardware manager
        dtype = self.precision

        hf_kwargs = {
            "torch_dtype": dtype,
            "trust_remote_code": self.trust_remote_code,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
            "low_cpu_mem_usage": True,
        }
        
        # If we have a specific device, try to map directly to it
        hf_kwargs["device_map"] = {"": self.device}

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            hf_kwargs["token"] = hf_token

        logger.info(
            "Cargando el embedding LLM '%s' desde cache=%s en %s con precision %s.",
            self.repo,
            self.cache_dir,
            self.device,
            dtype,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(self.repo, **hf_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModel.from_pretrained(self.repo, **hf_kwargs)
        # self._model.to(self.device) # Not needed with device_map
        self._model.eval()
        return self._tokenizer, self._model


    def encode(self, texts: List[str]) -> np.ndarray:
        tokenizer, model = self._ensure_model_and_tokenizer()

        texts_prefixed = self._apply_prefix(texts)
        n_total = len(texts_prefixed)
        n_batches = (n_total + self.batch_size - 1) // self.batch_size
        pool_fn = _last_token_pool if self.use_last_token_pool else _mean_pool
        pool_name = "last-token" if self.use_last_token_pool else "mean"

        logger.info(
            "Codificando %s textos con '%s' en %s lotes usando pooling %s.",
            n_total,
            self.name,
            n_batches,
            pool_name,
        )

        all_embeddings, start_idx = self._load_partial(texts)

        with torch.no_grad():
            for batch_idx, start in enumerate(
                range(start_idx, n_total, self.batch_size),
                start=start_idx // self.batch_size + 1,
            ):
                end = min(start + self.batch_size, n_total)
                batch = texts_prefixed[start:end]

                if n_batches <= 20 or batch_idx % 5 == 0 or batch_idx == n_batches:
                    logger.info(
                        "  [%s] lote %s/%s cubriendo instancias %s-%s.",
                        self.name,
                        batch_idx,
                        n_batches,
                        start,
                        end,
                    )

                encoded = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                outputs = model(**encoded)
                pooled = pool_fn(outputs.last_hidden_state, encoded["attention_mask"])
                pooled = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(pooled.cpu().float().numpy())

                if batch_idx % 5 == 0:
                    self._save_partial(all_embeddings, texts)

        result = np.vstack(all_embeddings).astype(np.float32)
        self._delete_partial(texts)
        logger.info("Codificacion finalizada para '%s'. Matriz obtenida: %s.", self.name, result.shape)
        return result

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
