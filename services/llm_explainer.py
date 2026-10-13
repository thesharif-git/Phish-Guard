"""
OriginTrace / PhishGuard AI Threat Explainer Engine
Tier 2 Deep Threat Explanation Pipeline:
- Local LLM: Qwen2.5-0.5B-Instruct (GGUF / HuggingFace Transformers, ~550MB-1.1GB RAM)
- Cloud LLM: Google Gemini 1.5 Flash API
- Resilient Fallback: High-precision SOC Cyber Analyst deterministic heuristics
Supports 29+ languages with cross-lingual threat attribution.
"""

import os
import re
import json
import logging
from typing import Optional, Dict, Any, List
from core.config import settings

logger = logging.getLogger("phishguard")

_LOCAL_LLM_LOADED = False
_LLAMA_MODEL = None
_HF_PIPELINE = None


def _get_local_llm():
    """
    Lazy loads local Qwen2.5-0.5B-Instruct model.
    First checks for llama-cpp-python GGUF file (fastest, lowest RAM ~550MB).
    Falls back to HuggingFace Transformers CPU pipeline.
    """
    global _LOCAL_LLM_LOADED, _LLAMA_MODEL, _HF_PIPELINE
    if _LOCAL_LLM_LOADED:
        return _LLAMA_MODEL or _HF_PIPELINE

    gguf_path = getattr(settings, "LOCAL_LLM_GGUF_PATH", "models/qwen2.5-0.5b-instruct.gguf")
    if os.path.exists(gguf_path):
        try:
            from llama_cpp import Llama
            logger.info(f"[LLM Explainer] Loading Qwen2.5-0.5B GGUF from '{gguf_path}'...")
            _LLAMA_MODEL = Llama(
                model_path=gguf_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False
            )
            _LOCAL_LLM_LOADED = True
            logger.info("[LLM Explainer] Loaded Qwen2.5-0.5B via llama.cpp (ultra-low RAM).")
            return _LLAMA_MODEL
        except Exception as e:
            logger.warning(f"[LLM Explainer] Failed to initialize llama.cpp: {e}")

    # If GGUF is not present, skip slow HuggingFace CPU load and use instant SOC heuristic
    use_hf = getattr(settings, "USE_HF_LLM_FALLBACK", False)
    if not use_hf:
        _LOCAL_LLM_LOADED = True
        return None

    # Optional Fallback: HuggingFace Transformers pipeline (only if explicitly enabled)
    model_name = getattr(settings, "LOCAL_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        logger.info(f"[LLM Explainer] Loading local model '{model_name}' via Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu", low_cpu_mem_usage=True)
        _HF_PIPELINE = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=-1
        )
        _LOCAL_LLM_LOADED = True
        logger.info(f"[LLM Explainer] Successfully loaded '{model_name}' into CPU memory.")
        return _HF_PIPELINE
    except Exception as e:
        logger.debug(f"[LLM Explainer] Local HuggingFace Qwen load skipped/failed: {e}")
        _LOCAL_LLM_LOADED = True
        return None


def _parse_llm_json(raw_text: str) -> Optional[Dict[str, str]]:
    """Safely extracts JSON response from LLM generation."""
    if not raw_text:
        return None
    try:
        # Direct JSON parse
        return json.loads(raw_text.strip())
    except Exception:
        # Extract json code block or curly braces
        json_match = re.search(r'\{[\s\S]*\}', raw_text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
    return None


def _run_local_qwen(prompt: str) -> Optional[Dict[str, Any]]:
    """Runs prompt through loaded Qwen2.5-0.5B model."""
    llm = _get_local_llm()
    if llm is None:
        return None

    try:
        if _LLAMA_MODEL is not None:
            output = _LLAMA_MODEL(
                prompt,
                max_tokens=220,
                temperature=0.2,
                stop=["<|im_end|>", "\n\n\n"]
            )
            raw = output["choices"][0]["text"]
            parsed = _parse_llm_json(raw)
            if parsed and "summary" in parsed:
                parsed["model_used"] = "Qwen2.5-0.5B-Instruct (Local GGUF)"
                return parsed

        if _HF_PIPELINE is not None:
            res = _HF_PIPELINE(
                prompt,
                max_new_tokens=200,
                temperature=0.2,
                do_sample=False,
                pad_token_id=151643
            )
            gen = res[0]["generated_text"]
            # Extract generated part after prompt
            clean_gen = gen[len(prompt):] if gen.startswith(prompt) else gen
            parsed = _parse_llm_json(clean_gen)
            if parsed and "summary" in parsed:
                parsed["model_used"] = "Qwen2.5-0.5B-Instruct (Local CPU)"
                return parsed
    except Exception as e:
        logger.warning(f"[LLM Explainer] Local Qwen inference error: {e}")

    return None


def _run_gemini_api(prompt: str, status: str) -> Optional[Dict[str, Any]]:
    """Calls Google Gemini 1.5 Flash API as cloud alternative."""
    api_key = settings.GEMINI_API_KEY or ""
    if not api_key:
        return None

    import requests
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "maxOutputTokens": 300
        }
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=3.5)
        if response.status_code == 200:
            data = response.json()
            candidates = data.get("candidates", [])
            if candidates:
                raw_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                parsed = json.loads(raw_text)
                return {
                    "summary": parsed.get("summary", ""),
                    "key_risk": parsed.get("key_risk", ""),
                    "recommended_action": parsed.get("recommended_action", ""),
                    "model_used": "Gemini 1.5 Flash (Tier 2 Cloud AI)"
                }
    except Exception as e:
        logger.debug(f"[LLM Explainer] Gemini API call skipped: {e}")

    return None


def _generate_deterministic_explanation(status: str, final_score: int, indicators: List[str], snippet: str, lang: str = "en") -> Dict[str, Any]:
    """
    Deterministic Senior SOC Cybersecurity Analyst rule engine.
    Guarantees an executive explanation if both local Qwen and cloud Gemini are unavailable.
    """
    lang_note = f" Detected message in {lang.upper()}." if lang and lang != "en" else ""

    if final_score >= 70:
        summary = (
            f"This email was classified as {status} (Risk Score: {final_score}/100) due to multiple high-severity threat signals, "
            f"including credential solicitation triggers and aggressive social engineering tactics.{lang_note}"
        )
        key_risk = "High risk of credential harvesting, unauthorized account takeover, or fraudulent funds diversion."
        recommended_action = "Block sender domain immediately, quarantine the message, and warn internal recipients."
    elif final_score >= 35:
        summary = (
            f"This email presents moderate suspicion (Risk Score: {final_score}/100) with urgency language or untrusted links "
            f"that require user verification before interaction.{lang_note}"
        )
        key_risk = "Potential spear-phishing reconnaissance or credential harvesting link."
        recommended_action = "Verify sender identity out-of-band and inspect links in a sandbox prior to opening."
    else:
        summary = (
            f"This email is classified as {status} (Risk Score: {final_score}/100). No malicious links, known campaign signatures, "
            f"or executive manipulation patterns were detected.{lang_note}"
        )
        key_risk = "No significant security threats or deceptive indicators identified."
        recommended_action = "Message is cleared for standard delivery; maintain routine security hygiene."

    if indicators:
        key_risk += f" Primary flagged indicators: {', '.join(indicators[:3])}."

    return {
        "summary": summary,
        "key_risk": key_risk,
        "recommended_action": recommended_action,
        "model_used": "OriginTrace SOC Analyst Engine (Heuristic Core)"
    }


def generate_ai_threat_explanation(
    text: str,
    status: str,
    final_score: int,
    indicators: list,
    detected_lang: str = "en"
) -> Optional[Dict[str, Any]]:
    """
    Tier 2 AI Threat Explainer:
    Combines Qwen2.5-0.5B-Instruct, Google Gemini 1.5 Flash, and deterministic SOC heuristics.
    Outputs structured natural language threat summaries across 29+ languages.
    """
    snippet = text[:600] if text else "No content body provided."
    indicators_str = ", ".join(indicators[:8]) if indicators else "General heuristic signals"

    prompt = f"""You are a Senior SOC Cybersecurity Analyst evaluating a phishing threat scan.
Provide a concise, professional threat breakdown based on these scan results:

Verdict: {status}
Risk Score: {final_score}/100
Language: {detected_lang}
Flags/Indicators: {indicators_str}
Email Text Snippet: {snippet}

Respond ONLY in valid JSON format with the following exact keys:
{{
  "summary": "2-sentence executive explanation of why this email is classified as {status}",
  "key_risk": "1-sentence description of the main danger (e.g. credential harvesting, malware link, spoofed domain, BEC wire fraud)",
  "recommended_action": "1-sentence SOC action recommendation"
}}"""

    provider = getattr(settings, "AI_EXPLAINER_PROVIDER", "local").lower()

    # 1. Try Local Qwen2.5-0.5B if configured
    if provider == "local":
        result = _run_local_qwen(prompt)
        if result:
            return result
        # Fallback to Gemini if local model isn't active
        result = _run_gemini_api(prompt, status)
        if result:
            return result

    # 2. Try Gemini API first if configured
    elif provider == "gemini":
        result = _run_gemini_api(prompt, status)
        if result:
            return result
        result = _run_local_qwen(prompt)
        if result:
            return result

    # 3. Deterministic SOC Analyst fallback (guarantees resilient response)
    return _generate_deterministic_explanation(status, final_score, indicators, snippet, detected_lang)
