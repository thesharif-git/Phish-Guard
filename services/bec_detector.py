"""
PhishGuard BEC Engine (Business Email Compromise & Intent Detector)
DistilBERT / Transformer Intent Classification + Social Engineering Analysis
Detects zero-link, zero-attachment financial fraud and CEO imposter scams.
"""

import re
import logging
import math
from threading import Lock
from typing import Dict, Any, List

logger = logging.getLogger("phishguard")

_BEC_PIPELINE = None
_PIPELINE_LOADED = False
_PIPELINE_LOAD_LOCK = Lock()


from core.config import settings

# Unvalidated multilingual review cues; topics alone do not establish fraud.
BEC_PATTERNS = {
    "CEO Wire Fraud": [
        # English
        "wire transfer", "wire payment", "transfer funds", "bank details", "bank account",
        "process payment", "vendor invoice", "confidential transaction", "in a meeting",
        "cannot take calls", "cannot talk", "do not call", "send transfer confirmation",
        "urgent payment", "immediate transfer", "transfer today",
        # Spanish
        "transferencia bancaria", "transferir fondos", "datos bancarios", "pagar factura",
        "transacción confidencial", "en una reunión", "no puedo recibir llamadas",
        # French
        "virement bancaire", "transférer des fonds", "coordonnées bancaires", "facture fournisseur",
        "transaction confidentielle", "en réunion",
        # German
        "banküberweisung", "überweisung ausführen", "bankverbindung", "dringende zahlung",
        "vertrauliche transaktion",
        # Arabic & Urdu
        "تحويل بنكي", "تحويل أموال", "بيانات الحساب", "معاملة سرية",
        "بینک ٹرانسفر", "فنڈز ٹرانسفر", "بینک اکاؤنٹ کی تفصیلات"
    ],
    "Gift Card Scam": [
        # English
        "gift card", "apple gift card", "steam card", "amazon card", "google play card",
        "purchase cards", "send codes", "scratch off", "buy cards for team",
        # Spanish
        "tarjetas de regalo", "tarjeta de regalo", "comprar tarjetas", "enviar códigos",
        # French
        "cartes-cadeaux", "carte cadeau", "acheter des cartes", "envoyer les codes",
        # German
        "gutscheinkarten", "geschenkkarte", "karten kaufen", "codes senden",
        # Arabic & Urdu
        "بطاقات الهدايا", "بطاقة هدايا", "إرسال الأكواد",
        "گفٹ کارڈ", "کوڈز بھیجیں"
    ],
    "Payroll Scam": [
        # English
        "direct deposit", "update bank account", "payroll account", "routing number",
        "paycheck deposit", "change direct deposit", "update my bank details",
        # Spanish
        "depósito directo", "actualizar cuenta bancaria", "cuenta de nómina", "cambiar cuenta",
        # French
        "virement direct", "mettre à jour mon compte", "compte de paie", "modifier coordonnées bancaires",
        # German
        "gehaltskonto", "bankdaten aktualisieren", "änderung der bankverbindung",
        # Arabic & Urdu
        "تحديث الحساب البنكي", "حساب الراتب", "تغيير بيانات الحساب",
        "بینک اکاؤنٹ اپ ڈیٹ کریں", "تنخواہ کا اکاؤنٹ"
    ],
    "Executive Coercion": [
        # English
        "are you at your desk", "quick task for you", "need your urgent assistance",
        "discreet task", "keep this between us", "handle this confidentially",
        # Spanish
        "estás en tu escritorio", "tarea rápida para ti", "necesito tu ayuda urgente",
        "tarea discreta", "mantenlo confidencial",
        # French
        "êtes-vous à votre bureau", "tâche urgente", "besoin de votre aide",
        "restez discret", "gardez cela confidentiel",
        # German
        "bist du an deinem platz", "dringende aufgabe", "brauche deine hilfe", "unter uns halten",
        # Arabic & Urdu
        "هل أنت في مكتبك", "مهمة عاجلة", "أمر سري للغاية",
        "کیا آپ اپنی میز پر ہیں", "فوری کام", "یہ بات ہمارے درمیان رہے"
    ],
    "Credential & Identity Theft": [
        # English
        "verification code", "sms code", "2-factor", "two-factor", "mfa code", "otp",
        "passcode", "social security number", "ssn", "current password", "workstation access",
        "account termination", "security violation", "reply with your password", "confirm your ssn",
        # Spanish
        "código de verificación", "código sms", "dos factores", "contraseña actual", "número de seguro social",
        # French
        "code de vérification", "code sms", "deux facteurs", "mot de passe actuel", "numéro de sécurité sociale",
        # German
        "bestätigungscode", "sms-code", "zwei-faktor", "aktuelles passwort", "sozialversicherungsnummer",
        # Arabic & Urdu
        "رمز التحقق", "رمز التأكيد", "كلمة المرور الحالية", "رقم الضمان الاجتماعي",
        "تصدیقی کوڈ", "موجودہ پاس ورڈ"
    ]
}


def _get_bec_classifier():
    """
    Lazy-load the configured zero-shot model as a PyTorch CPU pipeline,
    with the configured fallback when local primary weights are unavailable.
    """
    if _PIPELINE_LOADED:
        return _BEC_PIPELINE

    with _PIPELINE_LOAD_LOCK:
        if _PIPELINE_LOADED:
            return _BEC_PIPELINE
        return _load_bec_classifier()


def _load_bec_classifier():
    """Initialize while holding the loader lock; preserve unavailable caching."""
    global _BEC_PIPELINE, _PIPELINE_LOADED

    primary_model = getattr(settings, "BEC_MODEL", "MoritzLaurer/DeBERTa-v3-xsmall-mnli")
    fallback_model = getattr(settings, "FALLBACK_BEC_MODEL", "typeform/distilbert-base-uncased-mnli")

    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
        def local_classifier(model_name):
            # Pass offline loading to from_pretrained, not inference kwargs.
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
            return pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=-1)
        try:
            _BEC_PIPELINE = local_classifier(primary_model)
            logger.info(f"[BEC Engine] Primary intent classifier '{primary_model}' loaded successfully.")
        except Exception:
            try:
                _BEC_PIPELINE = local_classifier(fallback_model)
                logger.info(f"[BEC Engine] Fallback intent classifier '{fallback_model}' loaded successfully.")
            except Exception:
                logger.info("[BEC Engine] Pre-trained weights unavailable locally; using unvalidated review rules.")
                _BEC_PIPELINE = None
        _PIPELINE_LOADED = True
    except Exception as e:
        logger.warning(f"[BEC Engine] Transformer model load fallback to rule engine: {e}")
        _BEC_PIPELINE = None
        _PIPELINE_LOADED = True

    return _BEC_PIPELINE


def predict_bec_intent(text: str, urls_found: int = 0) -> Dict[str, Any]:
    """
    Evaluates email text for Business Email Compromise (BEC) intent.
    Extremely crucial for emails with 0 links and 0 attachments (pure social engineering).
    """
    if not text or len(text.strip()) < 10:
        return {
            "is_bec": False,
            "bec_score": 0,
            "confidence": None,
            "transformer_available": False,
            "transformer_score": None,
            "score_kind": "unvalidated_review_score",
            "intent_category": "Clean",
            "signals": []
        }

    text_lower = text.lower()
    matched_signals = []
    matched_categories = []
    rule_score = 0

    # ── Rule-Based BEC Pattern Matching ──────────────────────────────────────
    # Single-word keywords: use \b word boundaries (ASCII/Latin-safe).
    # Multi-word phrases: use plain substring match — phrase context already
    #   implies boundaries and \w boundaries break on Arabic/Urdu script.
    def _kw_matches(kw: str, text: str) -> bool:
        if " " in kw or any(ord(c) > 127 for c in kw):
            # Multi-word or non-ASCII: substring search is sufficient.
            return kw in text
        return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))

    for category, keywords in BEC_PATTERNS.items():
        hits = [kw for kw in keywords if _kw_matches(kw, text_lower)]
        if hits:
            matched_categories.append(category)
            rule_score += len(hits) * 20
            matched_signals.append(f"{category} indicator: '{hits[0]}'")

    rule_score = min(rule_score, 85)

    # Special boost for zero-link emails containing executive pressure
    if urls_found == 0 and re.search(r'\b(?:urgent|confidential|wire)\b', text_lower):
        if matched_categories:
            rule_score = min(rule_score + 15, 95)
            matched_signals.append("Zero-link email with executive financial coercion detected.")

    # ── Transformer DistilBERT Intent Scoring ─────────────────────────────────
    transformer_score = 0.0
    top_intent = "Clean"
    transformer_used = False
    transformer_available = False
    transformer_error = None

    try:
        classifier = _get_bec_classifier()
        if classifier is not None:
            candidate_labels = [
                "fraudulent wire transfer request",
                "gift card scam",
                "fraudulent payroll bank account change request",
                "credential or identity theft",
                "legitimate business email"
            ]
            res = classifier(text[:500], candidate_labels)

            if not isinstance(res, dict) or not res.get('labels') or not res.get('scores') or len(res['labels']) != len(res['scores']):
                raise ValueError('Malformed BEC classifier response')
            if any(label not in candidate_labels for label in res['labels']):
                raise ValueError('Unknown BEC classifier label')
            if res and "labels" in res and "scores" in res:
                top_label = res["labels"][0]
                top_score = float(res["scores"][0])
                if any(not math.isfinite(float(s)) or not 0 <= float(s) <= 1 for s in res['scores']):
                    raise ValueError("Invalid BEC classifier score")
                transformer_available = True

                if top_label != "legitimate business email" and top_score > 0.45:
                    transformer_score = float(top_score)
                    transformer_used = True
                    top_intent = top_label.title()
                    matched_signals.append(
                        f"Intent classifier suggested '{top_intent}' (uncalibrated label score {top_score:.2f})"
                    )
    except Exception as e:
        logger.debug(f"[BEC Engine] DistilBERT inference skipped: {e}")
        transformer_error = str(e)

    # These are independent, uncalibrated review signals. An absent keyword
    # match cannot dilute an observed semantic signal, or vice versa.
    if transformer_used:
        final_bec_score = max(rule_score, int(round(transformer_score * 100)))
    else:
        final_bec_score = rule_score

    final_bec_score = min(final_bec_score, 100)
    is_bec = final_bec_score >= 45 or len(matched_categories) >= 2

    primary_category = matched_categories[0] if matched_categories else (top_intent if transformer_used else "Clean")

    return {
        "is_bec": is_bec,
        "bec_score": final_bec_score,
        "confidence": None,
        "score_kind": "unvalidated_review_score",
        "rule_score": rule_score,
        "transformer_available": transformer_available,
        "transformer_score": transformer_score if transformer_used else None,
        "transformer_error": transformer_error,
        "transformer_text_characters_analyzed": min(len(text), 500) if transformer_available else 0,
        "transformer_text_truncated": transformer_available and len(text) > 500,
        "intent_category": primary_category,
        "signals": matched_signals if matched_signals else ["No BEC manipulation signals detected."]
    }
