"""Retain authentication assertions without granting them receiver trust.

No current ingestion route supplies receiver-authenticated results. A field named
Authentication-Results in an uploaded message is still sender-controllable.
"""
import re
from email import policy
from email.parser import Parser


def untrusted_authentication_report(headers, spf=None, dkim=None, dmarc=None):
    claims = []
    parsed = Parser(policy=policy.default).parsestr(headers or "", headersonly=True)
    # Parse only this field, never arbitrary X-* fields or message bodies.
    for value in parsed.get_all("Authentication-Results", []):
        text = str(value)
        # Ignore comments so quoted/commented status strings are not assertions.
        previous = None
        while previous != text:
            previous = text
            text = re.sub(r"\([^()]*\)", "", text)
        fields = text.split(";")
        authserv_id = fields[0].strip()
        for field in fields[1:]:
            match = re.match(r"\s*(spf|dkim|dmarc)\s*=\s*(pass|fail|softfail|neutral|none|temperror|permerror)\b", field, re.I)
            if match:
                claims.append({"method": match[1].lower(), "result": match[2].upper(),
                               "authserv_id": authserv_id, "source": "untrusted_message_header"})
    for method, value in (("spf", spf), ("dkim", dkim), ("dmarc", dmarc)):
        if value and str(value).upper() != "UNKNOWN":
            claims.append({"method": method, "result": str(value).upper(),
                           "source": "unverified_supplied_status"})
    return {"spf": "UNKNOWN", "dkim": "UNKNOWN", "dmarc": "UNKNOWN",
            "available": False, "all_passed": False, "verified": False,
            "reported_results": claims, "reported_available": bool(claims),
            "verification_source": None,
            "reason": "Authentication assertions are unverified; trusted receiver validation is unavailable."}
