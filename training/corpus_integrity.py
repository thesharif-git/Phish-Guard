"""Conservative identity and split rules, independent of any prediction model."""

from __future__ import annotations

import hashlib
import ipaddress
import re
import unicodedata
from urllib.parse import urlsplit, urlunsplit

import tldextract

# Offline, bundled PSL including private suffixes: no network/cache changes on a run.
PSL = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None,
                          include_psl_private_domains=True)
UNRESERVED = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
ENCODED = re.compile(r"%([0-9a-fA-F]{2})")
SPACE = re.compile(r"\s+")
EMAIL_ADDRESS = re.compile(r"\b[\w.+-]{1,64}@[\w.-]{1,253}\.[\w-]{1,63}\b")
WEB_ADDRESS = re.compile(r"https?://\S+", re.IGNORECASE)
NUMBERS = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def email_identity(text: str) -> tuple[str, str]:
    # Identity matching is deliberately separate from model canonicalization.
    canonical = SPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()
    template = WEB_ADDRESS.sub("<url>", canonical.casefold())
    template = EMAIL_ADDRESS.sub("<email>", template)
    template = NUMBERS.sub("<number>", template)
    return canonical, "email-template:" + digest(template)


def normalize_percent(value: str) -> str:
    def replace(match: re.Match) -> str:
        char = chr(int(match[1], 16))
        return char if char in UNRESERVED else "%" + match[1].upper()
    return ENCODED.sub(replace, value)


def url_identity(value: str, domain_only: bool = False) -> tuple[str, str]:
    raw = value.strip()
    if not raw or any(c.isspace() or ord(c) < 32 for c in raw):
        raise ValueError("empty_or_whitespace_url")
    has_scheme = re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", raw) is not None
    parts = urlsplit(raw if has_scheme else "http://" + raw)
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError("unsupported_scheme")
    host = parts.hostname
    if not host or "%" in host:
        # Ambiguous encoded hosts require separate browser-aware review.
        raise ValueError("missing_or_encoded_host")
    try:
        host = host.encode("idna").decode("ascii").lower().rstrip(".")
        port = parts.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("invalid_host_or_port") from exc
    try:
        ipaddress.ip_address(host)
        group = "host:" + host
    except ValueError:
        if len(host) > 253 or any(not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", p)
                                  for p in host.split(".")):
            raise ValueError("invalid_dns_host")
        ext = PSL(host)
        root = ext.top_domain_under_public_suffix
        if not root:
            raise ValueError("unrecognized_public_suffix")
        group = "host:" + root
    if domain_only:
        if parts.username or parts.password or port or parts.query or parts.fragment or parts.path not in ("", "/"):
            raise ValueError("not_a_domain")
        return host, group
    host_for_url = "[" + host + "]" if ":" in host else host
    # Preserve credentials, path case, query order, and fragments. Do not collapse
    # dot segments or decode reserved separators without browser equivalence tests.
    credentials = parts.netloc.rsplit("@", 1)[0] + "@" if "@" in parts.netloc else ""
    netloc = credentials + host_for_url + (":" + str(port) if port else "")
    canonical = urlunsplit((parts.scheme.lower(), netloc,
                           normalize_percent(parts.path), normalize_percent(parts.query),
                           normalize_percent(parts.fragment)))
    return canonical, group


def split_for_group(group: str, seed: str = "origintrace-corpus-v1") -> str:
    value = int(digest(seed + ":" + group)[:16], 16) / 2**64
    if value < 0.60:
        return "train"
    if value < 0.70:
        return "calibration"
    if value < 0.80:
        return "validation"
    return "development_test"


def validate_source(source: dict) -> None:
    if source["role"] not in {"development", "regression", "review_only", "final_holdout"}:
        raise ValueError("Unknown source role")
    if source["modality"] not in {"email", "url", "domain"}:
        raise ValueError("Unknown modality")
    if source["role"] == "review_only":
        return
    expected = {"email": "email_phishing", "url": "url_phishing", "domain": "domain_phishing"}
    if source.get("label_scope") != expected[source["modality"]]:
        raise ValueError("Label scope does not match detection task: " + source["id"])
    if not source.get("label_map") or set(source["label_map"].values()) - {0, 1}:
        raise ValueError("Explicit binary label mapping required")
    if source["modality"] == "domain" and source.get("annotation_verified") is not True:
        raise ValueError("Domain annotations must be verified; URL labels cannot be inherited")
    if source["role"] == "final_holdout" and not (
        source.get("annotation_verified") is True and source.get("collection_period")
        and source.get("previously_used_for_development") is False
    ):
        raise ValueError("Final holdout needs verified annotations, dates, and an unused-source declaration")
