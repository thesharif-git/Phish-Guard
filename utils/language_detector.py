"""
Unvalidated language hints and configured phrase observations.

Script/profile hints do not establish model language coverage, calibrated
confidence, malicious intent, or multilingual detection accuracy.
"""

import re
from typing import Dict, Any, List

# Script detection regex ranges
ARABIC_URDU_REGEX = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')
DEVANAGARI_REGEX = re.compile(r'[\u0900-\u097F]')
CYRILLIC_REGEX = re.compile(r'[\u0400-\u04FF]')
CHINESE_REGEX = re.compile(r'[\u4E00-\u9FFF]')
JAPANESE_REGEX = re.compile(r'[\u3040-\u309F\u30A0-\u30FF]')
KOREAN_REGEX = re.compile(r'[\uAC00-\uD7AF]')

# European language stopword profiles for fast lexical identification
LANGUAGE_STOPWORDS = {
    "es": {
        "words": {"el", "la", "de", "que", "en", "los", "las", "por", "para", "con", "su", "sus", "una", "del", "cuenta"},
        "name": "Spanish"
    },
    "fr": {
        "words": {"le", "la", "de", "des", "du", "que", "dans", "pour", "avec", "votre", "vos", "compte", "une", "est"},
        "name": "French"
    },
    "de": {
        "words": {"der", "die", "das", "und", "in", "den", "von", "zu", "mit", "sich", "auf", "für", "ist", "konto", "ihr"},
        "name": "German"
    },
    "pt": {
        "words": {"o", "a", "de", "que", "do", "da", "em", "um", "para", "com", "não", "uma", "os", "sua", "conta"},
        "name": "Portuguese"
    },
    "it": {
        "words": {"il", "la", "di", "che", "e", "un", "in", "per", "una", "con", "del", "sono", "suo", "conto", "vostro"},
        "name": "Italian"
    },
    "en": {
        "words": {"the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not", "on", "with", "your", "account"},
        "name": "English"
    },
    "no": {"words": {"og", "ikke", "for", "til", "med", "din", "ditt", "du", "det", "er", "som", "en", "på"}, "name": "Norwegian"},
    "hu": {"words": {"és", "nem", "hogy", "egy", "az", "a", "van", "ön", "önnek", "kérjük", "vagy", "meg"}, "name": "Hungarian"},
}

# Multilingual Phishing Keywords across Top Threat Vectors
MULTILINGUAL_PHISHING_KEYWORDS = {
    "es": [
        "verifique su cuenta", "cuenta suspendida", "actividad inusual", "haga clic aquí",
        "caduca hoy", "confirme su identidad", "actualice su facturación", "acceso no autorizado",
        "acción inmediata", "su cuenta será cerrada", "tiempo limitado", "actúe ahora", "urgente",
        "contraseña caducada", "bloqueo de seguridad", "transferencia urgente", "iniciar sesión",
        "pago pendiente", "factura no pagada"
    ],
    "fr": [
        "vérifiez votre compte", "compte suspendu", "activité inhabituelle", "cliquez ici",
        "expire aujourd'hui", "confirmez votre identité", "mettez à jour vos informations",
        "accès non autorisé", "action immédiate", "votre compte sera clôturé", "temps limité",
        "agissez maintenant", "urgent", "mot de passe expiré", "alerte de sécurité", "virement bancaire",
        "facture impayée", "connexion requise"
    ],
    "de": [
        "konto bestätigen", "konto gesperrt", "ungewöhnliche aktivität", "hier klicken",
        "läuft heute ab", "identität bestätigen", "rechnungsdaten aktualisieren", "unbefugter zugriff",
        "sofortiges handeln", "konto wird geschlossen", "begrenzte zeit", "jetzt handeln", "dringend",
        "passwort abgelaufen", "sicherheitswarnung", "überweisung ausführen", "offene rechnung"
    ],
    "pt": [
        "verifique sua conta", "conta suspensa", "atividade incomum", "clique aqui",
        "expira hoje", "confirme sua identidade", "atualize seus dados", "acesso não autorizado",
        "ação imediata", "sua conta será bloqueada", "tempo limitado", "aja agora", "urgente",
        "senha expirada", "alerta de segurança", "pagamento pendente"
    ],
    "it": [
        "verifica il tuo account", "conto sospeso", "attività insolita", "clicca qui",
        "scade oggi", "conferma la tua identità", "aggiorna i tuoi dati", "accesso non autorizzato",
        "azione immediata", "il tuo conto verrà chiuso", "tempo limitato", "agisci ora", "urgente",
        "password scaduta", "avviso di sicurezza"
    ],
    "ar": [
        "تأكيد حسابك", "تم تعليق حسابك", "نشاط غير عادي", "اضغط هنا", "ينتهي اليوم",
        "تأكيد هويتك", "تحديث بيانات الدفع", "دخول غير مصرح", "إجراء فوري", "سيتم إغلاق حسابك",
        "وقت محدود", "تحرك الآن", "عاجل", "كلمة المرور منتهية", "تحذير أمني", "تحويل بنكي",
        "تسجيل الدخول", "فاتورة مستحقة"
    ],
    "ur": [
        "اکاؤنٹ کی تصدیق", "اکاؤنٹ معطل", "غیر معمولی سرگرمی", "یہاں کلک کریں", "آج ہی ختم ہو جائے گا",
        "شناخت کی تصدیق", "ادائیگی کی معلومات اپ ڈیٹ کریں", "غیر مجاز رسائی", "فوری کارروائی",
        "اکاؤنٹ بند کر دیا جائے گا", "محدود وقت", "ارجنٹ", "فوری", "پاس ورڈ ایکسپائر", "بینک ٹرانسفر"
    ],
    "ru": [
        "подтвердите учетную запись", "аккаунт заблокирован", "подозрительная активность", "нажмите здесь",
        "истекает сегодня", "подтвердите личность", "обновите платежные данные", "несанкционированный доступ",
        "срочно", "пароль истек", "предупреждение безопасности", "банковский перевод"
    ],
    "zh": [
        "验证您的账户", "账户已被暂停", "异常活动", "点击此处", "今日到期",
        "确认您的身份", "更新您的账单", "未经授权的访问", "立即行动", "紧急",
        "密码已过期", "安全警报", "银行转账"
    ]
}


def _hint(lang="und", name="Undetermined", *, script=None, method="insufficient_language_evidence"):
    return {"lang": lang, "name": name, "is_english": None if lang == "und" else lang == "en",
            "confidence": None, "validated": False, "script": script, "method": method}


def detect_language(text: str) -> Dict[str, Any]:
    """
    Rapidly determines language of input email text.
    Handles non-Latin scripts (Arabic, Urdu, Cyrillic, Chinese, etc.) and
    lexical profiling for European languages.
    """
    if not text or len(text.strip()) < 5:
        return _hint()

    # 1. Non-Latin Script Detection (highest precision)
    sample = text[:1500]
    total_len = len(sample)

    ar_matches = len(ARABIC_URDU_REGEX.findall(sample))
    if ar_matches > 5 and (ar_matches / total_len) > 0.05:
        # Check specific Urdu markers
        urdu_markers = ["ہیں", "کریں", "آپ", "تھا", "ہوں", "کے"]
        is_urdu = any(marker in sample for marker in urdu_markers)
        return _hint("ur" if is_urdu else "ar", "Urdu" if is_urdu else "Arabic",
                     script="Arabic", method="unvalidated_script_and_marker_hint")

    cy_matches = len(CYRILLIC_REGEX.findall(sample))
    if cy_matches > 5 and (cy_matches / total_len) > 0.05:
        return _hint(name="Cyrillic script (language undetermined)", script="Cyrillic", method="script_hint")

    if len(KOREAN_REGEX.findall(sample)) > 5:
        return _hint("ko", "Korean", script="Hangul", method="script_hint")

    if len(JAPANESE_REGEX.findall(sample)) > 2:
        return _hint("ja", "Japanese", script="Kana", method="script_hint")

    zh_matches = len(CHINESE_REGEX.findall(sample))
    if zh_matches > 4 and (zh_matches / total_len) > 0.04:
        # Check Japanese kana
        if len(JAPANESE_REGEX.findall(sample)) > 2:
            return _hint("ja", "Japanese", script="Kana", method="script_hint")
        return _hint(name="Han script (language undetermined)", script="Han", method="script_hint")

    dev_matches = len(DEVANAGARI_REGEX.findall(sample))
    if dev_matches > 5 and (dev_matches / total_len) > 0.05:
        return _hint(name="Devanagari script (language undetermined)", script="Devanagari", method="script_hint")

    # 2. Latin-Script Lexical Matching
    # Tokenize words (lowercase, alphabetic only)
    words = re.findall(r'[^\W\d_]+', sample.casefold())
    if not words:
        return _hint()

    word_set = set(words)
    best_lang = "und"
    best_score = 0
    total_tokens = len(words)

    for lang_code, lang_info in LANGUAGE_STOPWORDS.items():
        overlap = len(word_set & lang_info["words"])
        # Weight by frequency if possible
        hit_count = sum(1 for w in words if w in lang_info["words"])
        score = hit_count / max(total_tokens, 1)

        if score > best_score:
            best_score = score
            best_lang = lang_code

    if best_score < 0.08:
        return _hint()
    return _hint(best_lang, LANGUAGE_STOPWORDS[best_lang]["name"], method="unvalidated_lexical_profile")


def find_multilingual_phishing_signals(text: str, detected_lang: str) -> List[str]:
    """
    Extracts matched phishing indicators and urgency phrases for the detected language,
    as well as cross-lingual universal lures.
    """
    if not text:
        return []

    text_lower = text.lower()
    signals = []

    # Mixed-language messages and ambiguous script hints must not hide observed
    # phrases. A dictionary's language tag describes the configured phrase,
    # not independently validated language identification of the message.
    languages = sorted(MULTILINGUAL_PHISHING_KEYWORDS, key=lambda code: (code != detected_lang, code))
    for lang in languages:
        for kw in MULTILINGUAL_PHISHING_KEYWORDS[lang]:
            if kw in text_lower:
                signals.append(f"[{lang.upper()}] Configured phrase match: '{kw}' (not proof of phishing)")

    return signals
