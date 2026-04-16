from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config.experiment import DATASET_CONFIG
from src.utils.io_utils import stable_text_hash

logger = logging.getLogger("phishing_xai.data_loader")


LABEL_MAP = {
    "phishing": 1,
    "phish": 1,
    "spam": 1,
    "malicious": 1,
    "1": 1,
    "true": 1,
    "yes": 1,
    "legitimate": 0,
    "legit": 0,
    "ham": 0,
    "benign": 0,
    "0": 0,
    "false": 0,
    "no": 0,
}


@dataclass(frozen=True)
class DatasetBundle:
    dataframe: pd.DataFrame
    subject_column: str
    label_column: str
    dataset_fingerprint: str

    @property
    def texts(self) -> list[str]:
        return self.dataframe["subject"].tolist()

    @property
    def labels(self) -> list[int]:
        return self.dataframe["label"].tolist()

    @property
    def size(self) -> int:
        return len(self.dataframe)


def _clean_subject(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _find_column(columns: list[str], candidates: list[str], explicit_name: str | None) -> str:
    if explicit_name is not None:
        lower_map = {column.lower(): column for column in columns}
        if explicit_name.lower() in lower_map:
            return lower_map[explicit_name.lower()]
        raise ValueError(f"La columna explicita '{explicit_name}' no existe. Columnas disponibles: {columns}")

    lower_map = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"No se encontro ninguna columna candidata {candidates}. Columnas disponibles: {columns}")


def _build_dataset_fingerprint(df: pd.DataFrame) -> str:
    serialised = "\n".join(f"{row.subject}\t{row.label}" for row in df.itertuples(index=False))
    return stable_text_hash(serialised)[:16]


def load_dataset(
    path: Path,
    *,
    subject_column: str | None = None,
    label_column: str | None = None,
) -> DatasetBundle:
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset en {path}.")

    logger.info("Leyendo dataset desde '%s'.", path)
    raw_df = pd.read_csv(path, low_memory=False)
    raw_df.columns = [str(column).strip() for column in raw_df.columns]

    subject_col = _find_column(
        list(raw_df.columns),
        DATASET_CONFIG["default_subject_column_candidates"],
        subject_column,
    )
    label_col = _find_column(
        list(raw_df.columns),
        DATASET_CONFIG["default_label_column_candidates"],
        label_column,
    )

    df = raw_df[[subject_col, label_col]].copy()
    df.columns = ["subject", "label"]

    initial_rows = len(df)
    df = df.dropna(subset=["subject", "label"]).copy()
    if len(df) != initial_rows:
        logger.info(
            "Se eliminaron %s filas con asunto o etiqueta ausente antes de limpiar el texto.",
            initial_rows - len(df),
        )

    original_labels = df["label"].astype(str).str.strip().str.lower()
    df["label"] = original_labels.map(LABEL_MAP)
    invalid_mask = df["label"].isna()
    if invalid_mask.any():
        invalid_values = sorted(original_labels[invalid_mask].unique().tolist())
        logger.warning(
            "Se eliminaron %s filas con etiquetas no reconocidas: %s",
            int(invalid_mask.sum()),
            invalid_values,
        )
        df = df.loc[~invalid_mask].copy()

    df["subject"] = df["subject"].map(_clean_subject)
    empty_mask = df["subject"].eq("")
    if empty_mask.any():
        logger.warning(
            "Se eliminaron %s filas porque el asunto quedo vacio tras la limpieza.",
            int(empty_mask.sum()),
        )
        df = df.loc[~empty_mask].copy()

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)

    legit_count = int((df["label"] == 0).sum())
    phish_count = int((df["label"] == 1).sum())
    fingerprint = _build_dataset_fingerprint(df)

    logger.info("Dataset preparado correctamente.")
    logger.info("  Columna de asunto utilizada: %s", subject_col)
    logger.info("  Columna de etiqueta utilizada: %s", label_col)
    logger.info("  Numero final de instancias: %s", len(df))
    logger.info("  Legitimos: %s", legit_count)
    logger.info("  Phishing: %s", phish_count)
    lengths = df["subject"].str.len()
    logger.info(
        "  Longitud de asunto en caracteres:\n"
        "    media=%.2f | mediana=%.2f | std=%.2f | varianza=%.2f\n"
        "    min=%s | max=%s\n"
        "    p25=%.0f | p75=%.0f | p95=%.0f\n"
        "    asimetria=%.4f | curtosis=%.4f",
        float(lengths.mean()),
        float(lengths.median()),
        float(lengths.std()),
        float(lengths.var()),
        int(lengths.min()),
        int(lengths.max()),
        float(lengths.quantile(0.25)),
        float(lengths.quantile(0.75)),
        float(lengths.quantile(0.95)),
        float(lengths.skew()),
        float(lengths.kurt()),
    )
    logger.info("  Fingerprint reproducible del dataset preparado: %s", fingerprint)

    return DatasetBundle(
        dataframe=df,
        subject_column=subject_col,
        label_column=label_col,
        dataset_fingerprint=fingerprint,
    )
