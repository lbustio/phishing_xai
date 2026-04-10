from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import requests

from config.paths import PROJECT_ROOT, SECRETS_DIR

logger = logging.getLogger("phishing_xai.llm_explainer")

try:
    from langdetect import DetectorFactory, LangDetectException, detect
    DetectorFactory.seed = 42
except ImportError:
    detect = None
    LangDetectException = Exception

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    logger.warning("'tenacity' no esta instalado. El sistema funcionara sin reintentos automaticos.")

    def retry(*args, **kwargs):
        return lambda f: f

    def stop_after_attempt(*args, **kwargs):
        return None

    def wait_exponential(*args, **kwargs):
        return None

    def retry_if_exception_type(*args, **kwargs):
        return None


class NaturalLanguageExplainer:
    LANGUAGE_NAMES = {
        "es": "Spanish",
        "en": "English",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "ca": "Catalan",
        "nl": "Dutch",
        "ro": "Romanian",
        "pl": "Polish",
        "sv": "Swedish",
        "no": "Norwegian",
        "da": "Danish",
        "fi": "Finnish",
        "tr": "Turkish",
        "cs": "Czech",
        "sk": "Slovak",
        "hu": "Hungarian",
        "ru": "Russian",
        "uk": "Ukrainian",
        "bg": "Bulgarian",
        "el": "Greek",
        "ar": "Arabic",
        "he": "Hebrew",
        "hi": "Hindi",
        "bn": "Bengali",
        "ur": "Urdu",
        "fa": "Persian",
        "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese",
        "zh": "Chinese",
        "ja": "Japanese",
        "ko": "Korean",
        "vi": "Vietnamese",
        "id": "Indonesian",
        "ms": "Malay",
        "th": "Thai",
    }

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = Path(base_path or PROJECT_ROOT)
        self.secrets_dir = Path(os.environ.get("XAI_SECRETS_DIR", SECRETS_DIR))
        self.hf_token = self._load_secret("huggingface.txt", aliases=["huggingFace.txt"])
        self.groq_key = self._load_secret("groq.txt")
        self._local_pipeline = None
        self._local_model_id = "Qwen/Qwen2.5-1.5B-Instruct"

        self.engine = "local"
        if self.groq_key:
            self.engine = "groq"
            logger.info("Motor primario: Groq")
        elif self.hf_token:
            self.engine = "hf"
            logger.info("Motor secundario: Hugging Face Router")
        else:
            logger.info("Motor terciario: modo local")

    def _load_secret(self, filename: str, aliases: Optional[list[str]] = None) -> Optional[str]:
        candidate_names = [filename, *(aliases or [])]
        for name in candidate_names:
            path = self.secrets_dir / name
            try:
                value = path.read_text(encoding="utf-8").strip()
                if value:
                    return value
            except FileNotFoundError:
                continue
        return None

    def _detect_language(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        if not normalized:
            return "es"

        tokens = re.findall(r"[a-záéíóúñ]+", normalized, flags=re.IGNORECASE)
        if not tokens:
            return "es"

        spanish_markers = {
            "el", "la", "de", "para", "con", "hola", "cuenta", "solicitud",
            "aprobacion", "aprobación", "seguridad", "urgente", "banco",
            "correo", "verificar", "estimado", "usted", "adjunto",
        }
        english_markers = {
            "the", "your", "please", "account", "request", "approval",
            "security", "verify", "workday", "outstanding", "urgent",
            "read", "immediately", "immediatly", "will", "regret", "mr",
            "hello", "click", "review",
        }

        spanish_score = sum(token in spanish_markers for token in tokens)
        english_score = sum(token in english_markers for token in tokens)

        padded = f" {normalized} "
        spanish_score += 2 * sum(
            fragment in padded
            for fragment in (" por favor ", " haga clic ", " estimado ", " le escribimos ", " su cuenta ")
        )
        english_score += 2 * sum(
            fragment in padded
            for fragment in (" please ", " click here ", " your account ", " read this ", " you will ")
        )

        if any(char in normalized for char in "áéíóúñ¿¡"):
            spanish_score += 2
        if re.search(r"\b(mr|mrs|dear|hello|urgent|read|will)\b", normalized):
            english_score += 1

        return "en" if english_score > spanish_score else "es"

    def _build_explanation_prompt(self, subject: str, phishing_probability: float, features: list[dict]) -> str:
        language = self._detect_language(subject)
        response_language = "espanol" if language == "es" else "English"
        verdict = "phishing" if phishing_probability >= 0.5 else "legitimate"

        if features:
            if language == "es":
                feature_lines = "\n".join(
                    f'- "{f["word"]}": {f["impact"]}% | {"sube riesgo de phishing" if f.get("positive", True) else "reduce riesgo y favorece legitimidad"}'
                    for f in features
                )
            else:
                feature_lines = "\n".join(
                    f'- "{f["word"]}": {f["impact"]}% | {"raises phishing risk" if f.get("positive", True) else "reduces risk and supports legitimacy"}'
                    for f in features
                )
        else:
            feature_lines = "- No individual trigger words could be isolated." if language == "en" else "- No se pudieron aislar palabras individuales."

        return (
            "You are an explainable AI forensic analyst.\n"
            f"Respond strictly in {response_language}.\n\n"
            f"Subject: \"{subject}\"\n"
            f"Model verdict: {verdict}\n"
            f"Phishing probability: {phishing_probability*100:.1f}%\n"
            f"Word attributions:\n{feature_lines}\n\n"
            "Write 4 to 5 plain sentences, no bullets, no markdown. "
            "Cite the most important words and percentages explicitly. "
            "If the email looks legitimate, explain why the words reduce risk. "
            "If it looks phishing, explain why the words increase risk. "
            "End with one concrete recommendation for the user."
        )

    def _build_body_prompt(self, subject: str, is_phishing: bool) -> str:
        language = self._detect_language(subject)
        message_type = "phishing" if is_phishing else "legitimate"

        if language == "es":
            return (
                f"Escribe unicamente el cuerpo de un correo {'phishing' if is_phishing else 'legítimo'} de un parrafo corto para este asunto: '{subject}'. "
                "Debes responder en espanol. "
                "Si el asunto parece legitimo, el cuerpo debe sonar profesional y normal. "
                "Si el asunto parece phishing, el cuerpo debe sonar manipulador pero plausible. "
                "No incluyas asunto, encabezado ni explicaciones. Solo el cuerpo del mensaje."
            )

        return (
            f"Write only the body of a short {message_type} email for this subject: '{subject}'. "
            "You must respond in English. "
            "If the subject looks legitimate, make the body sound professional and normal. "
            "If the subject looks like phishing, make the body persuasive and plausible. "
            "Do not include headers or explanations. Return only the body."
        )

    def _detect_language(self, text: str) -> str:
        normalized = (text or "").strip()
        if not normalized:
            return "es"

        if detect is not None:
            try:
                detected = detect(normalized)
                if detected:
                    return detected.lower()
            except LangDetectException:
                logger.warning("No se pudo detectar automaticamente el idioma; se aplicara respaldo heuristico.")
            except Exception as exc:
                logger.warning("Fallo inesperado al detectar idioma: %s", exc)

        lowered = normalized.lower()
        tokens = re.findall(r"[a-z]+", lowered, flags=re.IGNORECASE)
        if not tokens:
            return "es"

        spanish_markers = {
            "el", "la", "de", "para", "con", "hola", "cuenta", "solicitud",
            "aprobacion", "seguridad", "urgente", "banco",
            "correo", "verificar", "estimado", "usted", "adjunto",
        }
        english_markers = {
            "the", "your", "please", "account", "request", "approval",
            "security", "verify", "workday", "outstanding", "urgent",
            "read", "immediately", "immediatly", "will", "regret", "mr",
            "hello", "click", "review",
        }

        spanish_score = sum(token in spanish_markers for token in tokens)
        english_score = sum(token in english_markers for token in tokens)

        padded = f" {lowered} "
        spanish_score += 2 * sum(
            fragment in padded
            for fragment in (" por favor ", " haga clic ", " estimado ", " le escribimos ", " su cuenta ")
        )
        english_score += 2 * sum(
            fragment in padded
            for fragment in (" please ", " click here ", " your account ", " read this ", " you will ")
        )

        if " por favor " in padded or " estimado " in padded:
            spanish_score += 1
        if re.search(r"\b(mr|mrs|dear|hello|urgent|read|will)\b", lowered):
            english_score += 1

        return "en" if english_score > spanish_score else "es"

    def _build_explanation_prompt(self, subject: str, phishing_probability: float, features: list[dict]) -> str:
        verdict = "phishing" if phishing_probability >= 0.5 else "legitimate"

        if features:
            feature_lines = "\n".join(
                f'- "{f["word"]}": {f["impact"]}% | {"sube riesgo de phishing" if f.get("positive", True) else "reduce riesgo y favorece legitimidad"}'
                for f in features
            )
        else:
            feature_lines = "- No se pudieron aislar palabras individuales."

        return (
            "Eres un analista forense de inteligencia artificial explicable.\n"
            "Responde estrictamente en espanol, incluso si el asunto original esta en otro idioma.\n\n"
            f"Asunto analizado: \"{subject}\"\n"
            f"Veredicto del modelo: {verdict}\n"
            f"Probabilidad de phishing: {phishing_probability*100:.1f}%\n"
            f"Atribuciones por palabra:\n{feature_lines}\n\n"
            "Redacta 4 o 5 oraciones corridas, sin vietas ni markdown. "
            "Menciona explicitamente las palabras mas importantes y sus porcentajes. "
            "Si el correo parece legitimo, explica por que las palabras reducen el riesgo. "
            "Si parece phishing, explica por que las palabras aumentan el riesgo. "
            "Termina con una recomendacion concreta para la persona usuaria."
        )

    def _build_body_prompt(self, subject: str, is_phishing: bool) -> str:
        language = self._detect_language(subject)
        message_type = "phishing" if is_phishing else "legitimate"
        language_name = self.LANGUAGE_NAMES.get(language, language.upper())

        return (
            "You are generating a synthetic email body for controlled security research.\n"
            f"Write only in {language_name}. The output language must match the subject language exactly.\n"
            f"Subject: '{subject}'\n"
            f"Requested style: {message_type}\n"
            "Return only one short paragraph for the body of the email. "
            "Do not include the subject line, greeting metadata, headers, signatures, bullets, or explanations. "
            "If the subject appears legitimate, make the body sound professional and normal. "
            "If the subject appears phishing, make the body sound manipulative but plausible."
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((requests.exceptions.RequestException, Exception)),
        reraise=True,
    )
    def _call_groq(self, prompt: str) -> str:
        from groq import Groq

        client = Groq(api_key=self.groq_key)
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=350,
        )
        return chat_completion.choices[0].message.content.strip()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def _call_huggingface(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 350,
            "temperature": 0.3,
        }
        response = requests.post(
            "https://router.huggingface.co/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25,
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()

    def _generate_local(self, prompt: str) -> str:
        if self._local_pipeline is None:
            import torch
            from transformers import pipeline

            self._local_pipeline = pipeline(
                "text-generation",
                model=self._local_model_id,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )

        messages = [{"role": "user", "content": prompt}]
        outputs = self._local_pipeline(messages, max_new_tokens=180, temperature=0.3)
        return outputs[0]["generated_text"][-1]["content"].strip()

    def _run_prompt(self, prompt: str) -> str:
        if self.groq_key:
            try:
                return self._call_groq(prompt)
            except Exception as exc:
                logger.error("Groq fallo, usando fallback: %s", exc)

        if self.hf_token:
            try:
                return self._call_huggingface(prompt)
            except Exception as exc:
                logger.error("Hugging Face fallo, usando fallback local: %s", exc)

        return self._generate_local(prompt)

    def generate_explanation(self, subject: str, phishing_probability: float, lime_features: list[dict]) -> str:
        try:
            return self._run_prompt(self._build_explanation_prompt(subject, phishing_probability, lime_features))
        except Exception as exc:
            logger.error("No se pudo generar la explicacion natural: %s", exc)
            return "No se pudo generar el razonamiento natural con los motores disponibles."

    def generate_email_body(self, subject: str, is_phishing: bool) -> str:
        try:
            return self._run_prompt(self._build_body_prompt(subject, is_phishing))
        except Exception as exc:
            logger.error("No se pudo generar el cuerpo del mensaje: %s", exc)
            return "Could not generate a reconstructed message body." if self._detect_language(subject) == "en" else "No se pudo reconstruir el cuerpo del mensaje."

    def generate_fake_body(self, subject: str) -> str:
        return self.generate_email_body(subject, is_phishing=True)

    def process_batch(self, subjects: list[str], lime_keywords_batch: list[list[str]], shap_keywords_batch: list[list[str]], output_file: Path):
        results = []
        for subject in subjects:
            explanation = self.generate_explanation(subject, 0.95, [])
            results.append({"subject": subject, "natural_explanation": explanation})
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
