from __future__ import annotations

import logging
import multiprocessing
import platform

import torch

logger = logging.getLogger("phishing_xai.device")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
    except AttributeError:
        pass
    return torch.device("cpu")


def get_device_report() -> dict:
    device = get_device()
    report = {
        "device": str(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        report["device_name"] = props.name
        report["vram_total_gb"] = round(props.total_memory / (1024 ** 3), 2)
        report["cuda_capability"] = f"{props.major}.{props.minor}"
    elif device.type == "mps":
        report["device_name"] = "Apple Silicon (MPS)"
    else:
        report["device_name"] = "CPU"
        report["cpu_cores"] = multiprocessing.cpu_count()
    return report


def log_device_info(log: logging.Logger | None = None) -> torch.device:
    log = log or logger
    report = get_device_report()
    device = get_device()

    log.info("Resumen de hardware disponible para la corrida:")
    log.info("  Dispositivo seleccionado: %s (%s)", report["device_name"], report["device"])
    log.info("  Plataforma: %s", report["platform"])
    log.info("  Version de Python: %s", report["python"])
    log.info("  Version de PyTorch: %s", report["torch"])
    if "vram_total_gb" in report:
        log.info("  VRAM total detectada: %.2f GB", report["vram_total_gb"])
        log.info("  Capacidad CUDA: %s", report["cuda_capability"])
    if "cpu_cores" in report:
        log.info("  Nucleos CPU detectados: %s", report["cpu_cores"])
    if device.type == "cpu":
        log.warning(
            "La ejecucion ira por CPU. El pipeline sigue siendo valido, pero algunos "
            "embeddings de 7B se omitiran por coste computacional salvo que cambies la configuracion."
        )
    return device


def should_skip_model(model_config: dict, device: torch.device) -> tuple[bool, str]:
    if model_config.get("skip_on_cpu", False) and device.type == "cpu":
        return True, (
            "el modelo esta marcado como inviable en CPU por coste o memoria; "
            "se omite para evitar bloqueos de horas o errores de memoria"
        )

    if model_config.get("requires_hf_token", False):
        import os

        if not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
            return True, (
                "el modelo requiere token de Hugging Face y no se encontro ni HF_TOKEN "
                "ni HUGGING_FACE_HUB_TOKEN en el entorno"
            )

    return False, ""
