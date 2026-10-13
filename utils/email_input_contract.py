"""Shared, offline Email evidence input contract for research and inference.

The contract preserves supplied content and makes missing or untrusted evidence
explicit. It does not classify messages, contact destinations, verify reported
authentication, or establish a receiver observation time.
"""

import copy
import hashlib
import ipaddress
import json
import math
import re
import unicodedata
from email import policy
from email.parser import Parser
from urllib.parse import urlsplit, urlunsplit

from utils.email_authentication import untrusted_authentication_report
from utils.text_sanitizer import CONTROL_CHARS


EMAIL_INPUT_CONTRACT_VERSION = "email-evidence-v1"
_RAW_FIELDS = ("body", "subject", "sender", "headers", "html_body",
               "supplementary_urls", "parsing_info")
_PARSING_KEYS = {"version", "text_complete", "notes", "errors", "unscanned_parts",
                 "mime_defects", "source_kind"}
_UNSCANNED_PART_KEYS = {"filename", "content_type"}
# A scheme starts at a boundary. Without this assertion, a long plain alphabetic
# run retries the greedy scheme expression at every character (quadratic work).
_URL_CANDIDATE = re.compile(r"(?<![a-z0-9+.-])(?:[a-z][a-z0-9+.-]*://|www\.)[^\s<>\"']*", re.I)
_HTML_MARKUP = re.compile(
    r"<!--|<!doctype\b|</?(?:a|body|br|div|font|head|html|img|p|script|span|"
    r"style|table|tbody|td|template|th|tr|ul|li)\b[^>]*(?:>|$)", re.I)
_ANGLE_ADDRESS = re.compile(r"<\s*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\s*>")


def _require_json(value, path="parsing_info"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a nonfinite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _require_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{path} keys must be strings")
        for key, item in value.items():
            _require_json(item, f"{path}.{key}")
        return
    raise TypeError(f"{path} contains a non-JSON value")


def _normalized_view(raw):
    """Candidate-discovery view with offsets back to the supplied text.

    Full candidate strings are subsequently normalized as a whole. Character
    mapping is only used to find original slices for URL provenance.
    """
    pieces, positions = [], []
    for index, character in enumerate(raw):
        normalized = CONTROL_CHARS.sub("", unicodedata.normalize("NFKC", character))
        pieces.append(normalized)
        positions.extend([index] * len(normalized))
    return "".join(pieces), positions


def _destination(raw):
    value = CONTROL_CHARS.sub("", unicodedata.normalize("NFKC", raw)).strip()
    if any(character.isspace() for character in value):
        return None, "whitespace_in_url"
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ("http", "https"):
            return None, "unsupported_or_missing_scheme"
        hostname = parsed.hostname
        if not hostname:
            return None, "missing_hostname"
        port = parsed.port  # Reject malformed and out-of-range ports.
        try:
            ipaddress.ip_address(hostname)
            host = hostname.lower()
        except ValueError:
            host = hostname.encode("idna").decode("ascii").lower()
            labels = host.rstrip(".").split(".")
            if (len(host) > 253 or any(not label or len(label) > 63 or
                    not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) for label in labels)):
                return None, "invalid_hostname"
        if ":" in host:
            host = "[" + host + "]"
        userinfo = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
        netloc = userinfo + host + (":" + str(port) if port is not None else "")
        return urlunsplit((parsed.scheme.lower(), netloc, parsed.path, parsed.query, parsed.fragment)), None
    except (ValueError, UnicodeError):
        return None, "malformed_url"


def _header_assertions(headers, names):
    message = Parser(policy=policy.default).parsestr(headers, headersonly=True)
    return [{"value": str(value), "header": name,
             "source": "untrusted_message_header", "verified": False}
            for name in names for value in message.get_all(name, [])]


def build_email_input(body, *, subject="", sender="", headers="", html_body="",
                      supplementary_urls=None, parsing_info=None):
    """Build JSON-safe evidence from exact supplied Email fields.

    Uploaded/pasted headers remain assertions. Current ingestion routes do not
    provide trusted receiver authentication, so it is always absent here.
    """
    originals = {"body": body, "subject": subject, "sender": sender,
                 "headers": headers, "html_body": html_body}
    if any(not isinstance(value, str) for value in originals.values()):
        raise TypeError("Email body, subject, sender, headers and html_body must be strings")
    if supplementary_urls is not None and (not isinstance(supplementary_urls, list) or
            any(not isinstance(value, str) for value in supplementary_urls)):
        raise TypeError("supplementary_urls must be a list of strings or None")
    if parsing_info is not None and not isinstance(parsing_info, dict):
        raise TypeError("parsing_info must be a JSON object or None")
    _require_json(parsing_info)
    info = parsing_info or {}
    unknown_keys = set(info) - _PARSING_KEYS
    if unknown_keys:
        raise ValueError("Unsupported parsing_info fields: " + ", ".join(sorted(unknown_keys)))
    for key in ("version", "source_kind"):
        if key in info and not isinstance(info[key], str):
            raise TypeError(f"parsing_info.{key} must be a string")
    if "text_complete" in info and not isinstance(info["text_complete"], bool):
        raise TypeError("parsing_info.text_complete must be a boolean when supplied")
    for key in ("notes", "errors", "mime_defects"):
        if key in info and (not isinstance(info[key], list) or
                any(not isinstance(value, str) for value in info[key])):
            raise TypeError(f"parsing_info.{key} must be a list of strings")
    errors = info.get("errors", [])
    unscanned = info.get("unscanned_parts", [])
    if not isinstance(unscanned, list):
        raise TypeError("parsing_info.unscanned_parts must be a list")
    for part in unscanned:
        if not isinstance(part, dict) or any(not isinstance(value, str) for value in part.values()):
            raise TypeError("Each unscanned MIME part must contain string fields")
        if set(part) - _UNSCANNED_PART_KEYS:
            raise ValueError("Unsupported unscanned MIME part fields")

    links, unsupported, flags = [], [], []
    def add(raw, source, reference, anchor_text=None):
        destination, reason = _destination(raw)
        occurrence = {"raw_value": raw, "source": source,
                      "anchor_text": anchor_text, "reference": reference}
        if reason:
            unsupported.append(dict(occurrence, reason=reason))
        else:
            links.append(dict(occurrence, destination=destination,
                              normalization_applied=destination != raw))

    def scan_text(raw, source, reference):
        view, offsets = _normalized_view(raw)
        for match in _URL_CANDIDATE.finditer(view):
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            add(raw[start:end], source, dict(reference, start=start, end=end))

    def scan_html(markup, field):
        from bs4 import BeautifulSoup, Comment
        # Protect angle-addresses from tag interpretation. Exact input is retained
        # in the record; decoded text-node references do not claim raw byte offsets.
        protected = _ANGLE_ADDRESS.sub(lambda match: " " + match.group(1) + " ", markup)
        soup = BeautifulSoup(protected, "html.parser")
        excluded = soup(["script", "style", "head", "template"])
        if excluded:
            flags.append(field + ":noncontent_html_sections_ignored")
        for element in excluded:
            element.decompose()
        for index, anchor in enumerate(soup.find_all("a", href=True)):
            add(str(anchor["href"]), field + "_html_anchor",
                {"field": field, "kind": "decoded_html_anchor", "index": index},
                anchor.get_text(" ", strip=True))
        for index, node in enumerate(soup.find_all(string=True)):
            if isinstance(node, Comment):
                continue
            # An anchor's displayed URL is not its clickable destination. Keep
            # it as anchor_text for disagreement evidence; scan only its href.
            if any(parent.name == 'a' and parent.has_attr('href') for parent in node.parents):
                continue
            scan_text(str(node), field + "_visible_html_text",
                      {"field": field, "kind": "decoded_html_text_node", "index": index})

    normalized_body, _ = _normalized_view(body)
    if _HTML_MARKUP.search(normalized_body):
        scan_html(body, "body")
    else:
        scan_text(body, "body_text", {"field": "body", "kind": "original_text"})
    if html_body:
        scan_html(html_body, "html_body")
    for index, raw in enumerate(supplementary_urls or []):
        add(raw, "supplementary_url", {"field": "supplementary_urls", "index": index})

    auth = untrusted_authentication_report(headers)
    reply_to = _header_assertions(headers, ["Reply-To"])
    envelope_from = _header_assertions(headers, ["Return-Path", "Envelope-From", "X-Envelope-From"])
    completeness = ("INCOMPLETE" if errors or unscanned or info.get("text_complete") is False
                    else "COMPLETE" if info.get("text_complete") is True else "UNKNOWN")
    availability = {key: bool(value.strip()) for key, value in originals.items()}
    availability.update(supplementary_urls=bool(supplementary_urls), reply_to=bool(reply_to),
        envelope_from=bool(envelope_from), links=bool(links),
        reported_authentication=bool(auth["reported_available"]), trusted_authentication=False,
        parsing_info=parsing_info is not None)
    return dict(originals, contract_version=EMAIL_INPUT_CONTRACT_VERSION,
        supplementary_urls=copy.deepcopy(supplementary_urls), parsing_info=copy.deepcopy(parsing_info),
        reply_to=reply_to, envelope_from=envelope_from, links=links,
        link_destinations=list(dict.fromkeys(link["destination"] for link in links)),
        unsupported_urls=unsupported, authentication=auth, trusted_authentication=None,
        availability=availability,
        parsing={"completeness": completeness, "errors": copy.deepcopy(errors),
                 "unscanned_parts": copy.deepcopy(unscanned), "flags": flags})


def validate_email_input(record):
    """Require the same derived evidence as rebuilding from original fields."""
    if not isinstance(record, dict) or record.get("contract_version") != EMAIL_INPUT_CONTRACT_VERSION:
        raise ValueError("Unsupported Email input contract")
    _require_json(record, "record")
    if any(field not in record for field in _RAW_FIELDS):
        raise ValueError("Email input contract is missing original fields")
    rebuilt = build_email_input(record["body"], **{field: record[field] for field in _RAW_FIELDS if field != "body"})
    # JSON comparison also distinguishes true/false from 1/0, unlike Python
    # container equality. Derived schema types must match, not merely values.
    if _canonical_json(record) != _canonical_json(rebuilt):
        raise ValueError("Email input contract differs from its original evidence")
    return record


def _canonical_json(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def contract_sha256(record):
    """Hash validated evidence, independent of JSON key order or whitespace."""
    validate_email_input(record)
    encoded = _canonical_json(record).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evidence_sha256(record):
    """Dedup supplied evidence without parser notes or occurrence ordering.

    Keep headers and identity: identical wording with different sender/targets
    may represent opposite task labels. Template grouping handles those together.
    Full contract_sha256 remains the binding for annotations and inference.
    """
    validate_email_input(record)
    evidence = {field: record[field] for field in ('body', 'subject', 'sender', 'headers', 'html_body')}
    evidence['link_destinations'] = sorted(record['link_destinations'])
    evidence['unsupported_url_values'] = sorted(set(item['raw_value'] for item in record['unsupported_urls']))
    return hashlib.sha256(_canonical_json(evidence).encode('utf-8')).hexdigest()
