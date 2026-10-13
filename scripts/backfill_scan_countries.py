"""Fill missing hosting countries for historical flagged URL and domain scans."""

import argparse
import ipaddress
import socket
from urllib.parse import urlparse

import requests
from sqlmodel import Session, select

from data.database import engine
from data.models import ScanRecord


def host_from_scan(scan: ScanRecord) -> str:
    target = (scan.sender or "").strip()
    if not target:
        return ""
    parsed = urlparse(target if "://" in target else f"//{target}")
    return (parsed.hostname or "").lower()


def country_for_host(host: str) -> str:
    try:
        ip = socket.gethostbyname(host)
        if not ipaddress.ip_address(ip).is_global:
            return ""
        response = requests.get(
            f"http://ip-api.com/json/{ip}",
            params={"fields": "status,countryCode"},
            timeout=3,
        )
        response.raise_for_status()
        payload = response.json()
        code = str(payload.get("countryCode") or "").upper()
        return code if payload.get("status") == "success" and len(code) == 2 and code.isalpha() else ""
    except (OSError, requests.RequestException, ValueError):
        return ""


def main(apply: bool) -> None:
    with Session(engine) as session:
        scans = session.exec(
            select(ScanRecord).where(
                ScanRecord.status.in_(["PHISHING", "SUSPICIOUS"]),
                ScanRecord.scan_type.in_(["url", "domain"]),
            )
        ).all()
        missing = [scan for scan in scans if not (scan.full_analysis.get("geo") or {}).get("countryCode")]
        print(f"Historical scans missing country: {len(missing)}")
        if not apply:
            return

        cache = {}
        updated = 0
        for scan in missing:
            host = host_from_scan(scan)
            if not host:
                continue
            if host not in cache:
                cache[host] = country_for_host(host)
            code = cache[host]
            if not code:
                continue
            analysis = scan.full_analysis
            analysis["geo"] = {"countryCode": code, "source": "historical_ip_lookup"}
            scan.full_analysis = analysis
            session.add(scan)
            updated += 1
        session.commit()
        print(f"Verified countries saved: {updated}; unresolved: {len(missing) - updated}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Perform the authorized IP country lookups and save results")
    main(parser.parse_args().apply)
