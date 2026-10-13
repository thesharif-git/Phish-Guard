"""Research Email text representation; never used by the deployed scorer.

This transform is a candidate for a controlled representation ablation. It
does not establish detection quality or approve replacement training data.
Original MIME, destination URLs and identity evidence must remain separately
available; canonical text cannot replace that evidence.
"""

import html
import re
import unicodedata
from typing import Optional

from utils.text_sanitizer import (
    AMOUNT_REGEX,
    CONTROL_CHARS,
    EMAIL_REGEX,
    PHONE_REGEX,
    URL_REGEX,
)


EMAIL_CANDIDATE_TEXT_VERSION = "1.0.1"

# Recognize actual HTML without treating an RFC angle-address, <URL> token,
# comparison operator, or plain angle-delimited label as an HTML element.
_HTML_MARKUP = re.compile(
    r"<!--|<!doctype\b|</?(?:a|abbr|address|article|aside|b|blockquote|body|br|"
    r"button|caption|center|code|dd|del|div|dl|dt|em|fieldset|figcaption|figure|"
    r"font|footer|form|h[1-6]|head|header|hr|html|i|iframe|img|input|label|li|"
    r"link|main|meta|nav|ol|p|pre|script|section|small|span|strong|style|sub|sup|"
    r"table|tbody|td|template|textarea|th|thead|title|tr|u|ul|wbr)\b[^>]*(?:>|$)",
    re.IGNORECASE,
)
_ANGLE_ADDRESS = re.compile(
    r"<\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\s*>"
)
_ENTITY_TOKEN = re.compile(r"<(?:URL|EMAIL|PHONE|AMOUNT)>", re.IGNORECASE)


def canonical_email_candidate_text(text: Optional[str]) -> str:
    """Normalize before masking and extract visible text with link presence.

    HTML hrefs contribute a generic URL token beside their visible anchor text,
    rather than leaking a URL identity into the content representation. Source
    names and dates are preserved: there is no dataset-specific blacklist.
    """
    if not isinstance(text, str) or not text:
        return ""

    normalized = unicodedata.normalize("NFKC", html.unescape(text))
    # Some C0 controls are whitespace separators, not zero-width obfuscation.
    # Removing a vertical tab/form feed joins adjacent words and can synthesize
    # a URL. Keep its boundary; continue removing genuinely nonspacing controls.
    normalized = CONTROL_CHARS.sub(lambda match: " " if match.group(0).isspace() else "", normalized)
    # A sender address is content in both plain text and mixed HTML inputs.
    normalized = _ANGLE_ADDRESS.sub(lambda match: " " + match.group(1) + " ", normalized)

    if _HTML_MARKUP.search(normalized):
        from bs4 import BeautifulSoup, NavigableString

        # Existing mask tokens are literal text, not custom HTML elements.
        # Escape them for this parse; get_text restores their angle brackets.
        markup = _ENTITY_TOKEN.sub(lambda match: html.escape(match.group(0)), normalized)
        soup = BeautifulSoup(markup, "html.parser")
        for element in soup(["script", "style", "head", "template"]):
            element.decompose()
        for anchor in soup.find_all("a", href=True):
            # Normalize happened before parsing, including obfuscated schemes.
            # Do not count non-URL hrefs, such as local fragments, as destinations.
            if URL_REGEX.fullmatch(str(anchor["href"]).strip()):
                anchor.insert_before(NavigableString(" <URL> "))
        normalized = soup.get_text(" ", strip=True)

    masked = URL_REGEX.sub(" <URL> ", normalized)
    masked = EMAIL_REGEX.sub(" <EMAIL> ", masked)
    masked = PHONE_REGEX.sub(" <PHONE> ", masked)
    masked = AMOUNT_REGEX.sub(" <AMOUNT> ", masked)
    return re.sub(r"\s+", " ", masked).strip().lower()
