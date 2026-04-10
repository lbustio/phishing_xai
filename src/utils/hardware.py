import torch
import psutil
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("phishing_xai.utils.hardware")

class HardwareManager:
    """Manages hardware detection and adaptive model selection."""
    
    @staticmethod
    def get_system_stats() -> Dict[str, Any]:
        """Returns current system resource stats."""
        mem = psutil.virtual_memory()
        stats = {
            "ram_total_gb": mem.total / (1024**3),
            "ram_available_gb": mem.available / (1024**3),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "vram_total_gb": 0.0
        }
        
        if stats["cuda_available"]:
            # Get VRAM of the first device
            stats["vram_total_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
        return stats

    @staticmethod
    def get_model_requirements(model_name: str) -> Dict[str, Any]:
        """Returns estimated RAM/VRAM requirements for a model using heuristics and known lookups."""
        m_lower = model_name.lower()
        
        # 1. Known hard lookups (for precision)
        known = {
            "Salesforce/SFR-Embedding-Mistral": {"fp32_gb": 28.0, "fp16_gb": 14.0, "type": "llm"},
            "Salesforce/SFR-Embedding-2_R": {"fp32_gb": 28.0, "fp16_gb": 14.0, "type": "llm"},
            "BAAI/bge-m3": {"fp32_gb": 2.2, "fp16_gb": 1.1, "type": "encoder"},
            "sentence-transformers/all-mpnet-base-v2": {"fp32_gb": 0.6, "fp16_gb": 0.3, "type": "encoder"},
        }
        if model_name in known:
            return known[model_name]

        # 2. Heuristics for unknown models
        # LLMs (Usually 7B/8B)
        if "-7b" in m_lower or "-8b" in m_lower or "mistral" in m_lower or "llama" in m_lower:
            return {"fp32_gb": 28.0, "fp16_gb": 14.0, "type": "llm"}
        
        # Heavy Encoders (Large)
        if "large" in m_lower or "instructor-xl" in m_lower:
            return {"fp32_gb": 3.0, "fp16_gb": 1.5, "type": "encoder"}
            
        # Standard Encoders (Base / Multi)
        if "base" in m_lower or "multi" in m_lower or "medium" in m_lower:
            return {"fp32_gb": 1.5, "fp16_gb": 0.8, "type": "encoder"}

        # Light/Small Models
        if "small" in m_lower or "distil" in m_lower or "tiny" in m_lower or "mini" in m_lower:
            return {"fp32_gb": 0.5, "fp16_gb": 0.25, "type": "encoder"}

        # Default fallback for unknown
        return {"fp32_gb": 2.0, "fp16_gb": 1.0, "type": "unknown"}

    @classmethod
    def suggest_config(cls, target_model: str) -> Dict[str, Any]:
        """Suggests the best configuration for a target model based on local resources."""
        stats = cls.get_system_stats()
        reqs = cls.get_model_requirements(target_model)
        
        suggestion = {
            "original_model": target_model,
            "device": "cpu",
            "dtype": torch.float32,
            "low_cpu_mem_usage": True,
            "status": "ok",
            "message": "",
            "can_run": True
        }
        
        # 1. Device selection
        if stats["cuda_available"]:
            suggestion["device"] = "cuda"
            # If VRAM fits FP16, use it
            if stats["vram_total_gb"] >= reqs["fp16_gb"]:
                suggestion["dtype"] = torch.float16
            else:
                # If too big for GPU, fallback to CPU
                suggestion["device"] = "cpu"
                suggestion["message"] = f"VRAM ({stats['vram_total_gb']:.1f}GB) insuficiente para {target_model}. Usando CPU."
        
        # 2. Precision for CPU
        if suggestion["device"] == "cpu":
            # Adaptive Precision for CPU (bfloat16 preferred for modern CPUs)
            if stats["ram_total_gb"] <= 16.0:
                suggestion["dtype"] = torch.bfloat16
            else:
                suggestion["dtype"] = torch.float32
            
            # Check if it fits in available RAM (with buffer)
            required_ram = reqs["fp16_gb"] + 2.5 
            if stats["ram_available_gb"] < required_ram:
                suggestion["can_run"] = False
                suggestion["status"] = "error"
                suggestion["message"] = f"RAM insuficiente para {target_model} ({stats['ram_available_gb']:.1f}GB disponibles)."
        
        return suggestion
