import email
import re
from email import policy
from urllib.parse import urlsplit

EMAIL_PARSING_VERSION = "3.0"


def visible_html_text(markup: str) -> str:
    """Extract visible text offline; keep original HTML separately for evidence."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup(["script", "style", "head", "template"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def decode_email_text(payload: bytes, charset: str | None = None):
    """Use the declared codec, or strict UTF-8 when an undeclared ASCII decode fails.

    Never guess another codec for invalid declared bytes. Replacement/error
    details remain observable and cannot silently become complete evidence.
    """
    codec = charset or "ascii"
    notes, errors = [], []
    try:
        text = payload.decode(codec, errors="strict")
    except LookupError:
        text = payload.decode("utf-8", errors="replace")
        errors.append("unknown_declared_charset:" + str(charset))
    except UnicodeDecodeError:
        if not charset:
            try:
                text = payload.decode("utf-8", errors="strict")
                notes.append("recovered_utf8_without_declared_charset")
            except UnicodeDecodeError:
                text = payload.decode("ascii", errors="replace")
                errors.append("undeclared_non_ascii_non_utf8_payload")
        else:
            text = payload.decode(codec, errors="replace")
            errors.append("invalid_declared_charset_payload:" + str(charset))
    if "\ufffd" in text:
        errors.append("replacement_character_in_decoded_text")
    return text, notes, errors


def _message_parts(message):
    """Traverse outer message leaves; never select a nested attached email body."""
    if message.get_content_type() == "message/rfc822":
        yield message
    elif message.is_multipart():
        for child in message.iter_parts():
            if child.get_content_disposition() == "attachment" and child.is_multipart():
                yield child
            else:
                yield from _message_parts(child)
    else:
        yield message


def extract_email_anchor_links(html_body: str) -> list[dict]:
    """Preserve visible text and HTTP(S) hrefs without fetching destinations."""
    if not html_body:
        return []
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_body, "html.parser")
    links = []
    seen = set()
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"]).strip()
        try:
            parsed = urlsplit(href)
            if parsed.scheme.lower() not in ("http", "https") or not parsed.hostname:
                continue
            if any(c.isspace() or ord(c) < 32 for c in href):
                continue
        except ValueError:
            continue
        href = parsed.scheme.lower() + href[len(parsed.scheme):]
        text = anchor.get_text(" ", strip=True)
        key = (text, href)
        if key not in seen:
            seen.add(key)
            links.append({"text": text, "href": href})
    return links

def parse_eml_content(file_bytes: bytes) -> dict:
    """
    Parses a raw .eml file byte array into a structured dictionary containing
    the sender, subject, complete raw headers, and plain-text body.
    """
    msg = email.message_from_bytes(file_bytes, policy=policy.default)
    
    sender = msg.get('From', '')
    subject = msg.get('Subject', '')
    
    # Extract all headers as a single block of text for the security analyzer
    headers_list = []
    for k, v in msg.items():
        headers_list.append(f"{k}: {v}")
    raw_headers = "\n".join(headers_list)
    
    # get_body follows MIME alternative/related selection and excludes attached
    # messages. Single-part and multipart HTML now have the same text contract.
    attachments = []
    html_parts = []
    decoded = {}
    notes, errors, unscanned = [], [], []
    for part in _message_parts(msg):
        mime = part.get_content_type()
        filename = part.get_filename() or ""
        attached = part.get_content_disposition() == "attachment"
        if part.is_multipart() or mime == "message/rfc822":
            unscanned.append({"filename": filename, "content_type": mime})
            continue
        payload = part.get_payload(decode=True) or b""
        if mime in ("text/plain", "text/html") and not attached:
            text, recovered, failures = decode_email_text(payload, part.get_content_charset())
            decoded[id(part)] = text
            notes.extend(recovered)
            errors.extend(failures)
            if mime == "text/html":
                html_parts.append(text)
        elif mime.startswith("image/") or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif')):
            if payload:
                attachments.append({"filename": filename or "embedded_image.png", "content_type": mime, "data": payload})
            else:
                errors.append("empty_image_payload")
        elif attached or filename or mime not in ("text/plain", "text/html"):
            unscanned.append({"filename": filename, "content_type": mime})
    selected = msg.get_body(preferencelist=("plain", "html"))
    body = decoded.get(id(selected), "")
    if selected is not None and selected.get_content_type() == "text/html":
        body = visible_html_text(body)
    # An empty plain alternative must not hide a nonempty HTML alternative.
    if not body.strip():
        selected_html = msg.get_body(preferencelist=("html",))
        body = visible_html_text(decoded.get(id(selected_html), ""))
    # Transfer decoding populates Base64 defects lazily; inspect it afterwards.
    defects = sorted({type(d).__name__ for part in msg.walk() for d in part.defects})
    errors.extend("mime_defect:" + defect for defect in defects)
    if sum(part.get_content_type() == 'text/plain' and id(part) in decoded for part in _message_parts(msg)) > 1:
        errors.append('multiple_inline_plain_parts_require_review')

    return {
        "sender": str(sender),
        "subject": str(subject),
        "headers": raw_headers,
        "body": body.strip(),
        "html_body": "\n".join(html_parts),
        "attachments": attachments,
        "parsing": {"version": EMAIL_PARSING_VERSION, "text_complete": not errors,
                    "notes": sorted(set(notes)), "errors": sorted(set(errors)),
                    "unscanned_parts": unscanned, "mime_defects": defects,
                    "source_kind": "rfc_email"},
    }


def parse_pasted_email(content: str) -> dict | None:
    """Parse a complete pasted message only when it has a plausible RFC header block."""
    match = re.search(r"\r?\n\r?\n", content)
    if not match:
        return None
    header_lines = content[:match.start()].splitlines()
    if not 1 <= len(header_lines) <= 100:
        return None
    if not any(re.match(r"^(From|Subject|Received|Authentication-Results):", line, re.I) for line in header_lines):
        return None
    if not all(re.match(r"^[\x21-\x39\x3b-\x7e]+:", line) or line[:1].isspace() for line in header_lines):
        return None
    parsed = parse_eml_content(content.encode("utf-8"))
    return parsed if parsed["body"] or parsed["html_body"] else None
