import json
import logging
import requests
import os
from pathlib import Path
from typing import Optional, List, Dict

from config.paths import SECRETS_DIR

# Define logger first to enable robust error reporting during imports
logger = logging.getLogger("phishing_xai.llm_explainer")

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    # Robust fallback: No-op decorator if tenacity is missing in the host environment
    logger.warning("⚠️ 'tenacity' no está instalado. El sistema funcionará sin reintentos automáticos.")
    def retry(*args, **kwargs):
        return lambda f: f
    def stop_after_attempt(*args, **kwargs): return None
    def wait_exponential(*args, **kwargs): return None
    def retry_if_exception_type(*args, **kwargs): return None

class NaturalLanguageExplainer:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.secrets_dir = Path(os.environ.get("XAI_SECRETS_DIR", SECRETS_DIR))
        self.hf_token = self._load_secret("huggingface.txt", aliases=["huggingFace.txt"])
        self.groq_key = self._load_secret("groq.txt")
        
        # Local Engine State (Using Qwen for broad accessibility)
        self._local_pipeline = None
        self._local_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
        # Determine the best active engine
        self.engine = "hf" # Default fallback
        if self.groq_key:
            self.engine = "groq"
            logger.info("🚀 Motor de Inteligencia Primario: Groq (Ultra-Rápido)")
        elif self.hf_token:
            self.engine = "hf"
            logger.info("☁️ Motor de Inteligencia Secundario: HuggingFace Router")
        else:
            self.engine = "local"
            logger.info("🏠 Motor de Inteligencia Terciario: Modo Offline Forense (Local 1B)")

    def _load_secret(self, filename: str, aliases: Optional[List[str]] = None) -> Optional[str]:
        candidate_names = [filename, *(aliases or [])]
        candidate_paths = [self.secrets_dir / name for name in candidate_names]

        for path in candidate_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    value = f.read().strip()
                    if value:
                        return value
            except FileNotFoundError:
                continue

        return None

    def _get_prompt(self, subject: str, confidence: float, lime_features: list[dict]) -> str:
        if lime_features:
            features_block = "ATRIBUCIÓN CUANTITATIVA POR PALABRA (leave-one-out — eliminar esa palabra cambia la probabilidad en ese porcentaje):\n"
            for f in lime_features:
                direction = "SUBE el riesgo de phishing" if f.get("positive", True) else "BAJA el riesgo (señal de legitimidad)"
                features_block += f"  • \"{f['word']}\": {f['impact']}% — {direction}\n"
        else:
            features_block = "ATRIBUCIÓN: No se pudieron aislar palabras individuales (señal global del embedding).\n"

        return (
            "Eres el Sistema Maestro de Inteligencia Artificial Explicable (XAI) de Phishing PowerToy.\n\n"
            f"DATOS DEL ESCANEO:\n"
            f"  Asunto analizado: \"{subject}\"\n"
            f"  Probabilidad de phishing: {confidence*100:.1f}%\n\n"
            f"{features_block}\n"
            "INSTRUCCIÓN (OBLIGATORIO E INNEGOCIABLE):\n"
            "Escribe en español un análisis forense de 4 a 5 oraciones corridas (sin viñetas, sin negritas, solo texto plano). "
            "Cada oración DEBE hacer referencia directa a los datos numéricos anteriores. Específicamente:\n"
            "1. Cita CADA palabra de la lista con su porcentaje EXACTO y explica la táctica psicológica que usa "
            "(urgencia, miedo, autoridad, codicia, etc.). Ejemplo obligatorio del estilo: "
            "'El término \"financial\" concentra el 38% del riesgo porque activa el miedo a consecuencias económicas, "
            "táctica clásica de ingeniería social para forzar una acción impulsiva.'\n"
            "2. Explica cómo la COMBINACIÓN de esas palabras juntas amplifica el efecto más allá de cada una por separado.\n"
            "3. Describe cómo se vería un mensaje legítimo sobre el mismo tema (tono institucional, sin urgencia artificial).\n"
            "4. Da una recomendación forense concreta para este caso.\n\n"
            "PROHIBIDO: análisis genérico sin citar palabras y porcentajes. "
            "PROHIBIDO: introducciones del tipo 'El análisis indica que...'. Empieza directo con los términos detectados."
        )

    # --- ENGINE-SPECIFIC CALLS WITH TENACITY RETRIES ---

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        reraise=True
    )
    def _call_groq(self, prompt: str) -> str:
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("⚠️ El paquete 'groq' no está instalado en este entorno.")
            
        client = Groq(api_key=self.groq_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=300,
        )
        return chat_completion.choices[0].message.content.strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True
    )
    def _call_huggingface(self, prompt: str) -> str:
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.3
        }
        response = requests.post(api_url, headers=headers, json=payload, timeout=25)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        raise ValueError("Respuesta de HF no contiene el formato esperado.")

    # --- MAIN ENTRY POINT ---

    def generate_explanation(self, subject: str, confidence: float, lime_features: list[dict]) -> str:
        prompt = self._get_prompt(subject, confidence, lime_features)
        
        # 1. Attempt Groq
        if self.groq_key:
            try:
                return self._call_groq(prompt)
            except Exception as e:
                logger.error("Fallo definitivo en Groq tras reintentos, bajando a secundario: %s", e)
        
        # 2. Attempt HuggingFace
        if self.hf_token:
            try:
                return self._call_huggingface(prompt)
            except Exception as e:
                logger.error("Fallo definitivo en HuggingFace tras reintentos, bajando a local: %s", e)

        # 3. Final Fallback: Local Offline
        try:
            return self._generate_local(prompt)
        except Exception as e:
            logger.error("Fallo final en Modo Offline: %s", e)
            return "Detectado como phishing. (El análisis de síntesis falló en todos los motores)."

    def _generate_local(self, prompt: str) -> str:
        if self._local_pipeline is None:
            logger.info("🏠 Iniciando Modo Offline Forense... Cargando modelo local 1.5B (esto puede tardar unos segundos la primera vez)")
            import torch
            from transformers import pipeline
            self._local_pipeline = pipeline(
                "text-generation",
                model=self._local_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
        
        messages = [{"role": "user", "content": prompt}]
        outputs = self._local_pipeline(messages, max_new_tokens=150, temperature=0.3)
        return outputs[0]["generated_text"][-1]["content"].strip()

    def generate_fake_body(self, subject: str) -> str:
        """
        Generate a synthetic phishing email body for the given subject.
        Follows the same Groq → HuggingFace → Local fallback chain.
        """
        prompt = (
            f"Escribe únicamente el cuerpo de un email de phishing de 1 párrafo corto "
            f"para este asunto: '{subject}'. "
            "No incluyas asunto, encabezado ni explicaciones. Solo el cuerpo del mensaje."
        )

        # 1. Groq
        if self.groq_key:
            try:
                return self._call_groq(prompt)
            except Exception as e:
                logger.error("Fallo en Groq para cuerpo falso, bajando a secundario: %s", e)

        # 2. HuggingFace
        if self.hf_token:
            try:
                return self._call_huggingface(prompt)
            except Exception as e:
                logger.error("Fallo en HuggingFace para cuerpo falso, bajando a local: %s", e)

        # 3. Local
        try:
            return self._generate_local(prompt)
        except Exception as e:
            logger.error("Fallo en modo local para cuerpo falso: %s", e)
            return "No se pudo reconstruir el cuerpo del mensaje."

    def process_batch(self, subjects: list[str], lime_keywords_batch: list[list[str]], shap_keywords_batch: list[list[str]], output_file: Path):
        results = []
        for i in range(len(subjects)):
            explanation = self.generate_explanation(subjects[i], 0.95, [])
            results.append({"subject": subjects[i], "natural_explanation": explanation})
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
