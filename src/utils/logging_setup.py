from __future__ import annotations

import io
import logging
import sys
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class _FlushingStreamHandler(logging.StreamHandler):
    """StreamHandler that flushes after every record for real-time visibility in IDEs/panels."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


def setup_pipeline_logger(log_file: Path, logger_name: str = "phishing_xai") -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Force write-through on stdout so messages appear immediately when piped
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(write_through=True)
        except Exception:
            pass

    console_handler = _FlushingStreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False

    for noisy_logger in [
        "matplotlib",
        "PIL",
        "urllib3",
        "transformers",
        "sentence_transformers",
    ]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"phishing_xai.{name}")


def log_section(logger: logging.Logger, title: str) -> None:
    logger.info("=" * 68)
    logger.info(title)
    logger.info("=" * 68)


def log_subsection(logger: logging.Logger, title: str) -> None:
    logger.info("-" * 68)
    logger.info(title)
    logger.info("-" * 68)
