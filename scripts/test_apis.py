"""Quick verification: run with .venv\Scripts\python.exe test_apis.py"""
import os, sys
# Windows console encoding fix
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Fix broken SSLKEYLOGFILE before importing requests
_ssl = os.environ.get("SSLKEYLOGFILE", "")
if _ssl.startswith('"') or _ssl.startswith("'"):
    del os.environ["SSLKEYLOGFILE"]
    print(f"[FIX] Removed malformed SSLKEYLOGFILE")

from dotenv import load_dotenv
load_dotenv(override=True)
import requests, base64

key = os.getenv("VIRUSTOTAL_API_KEY", "").strip()
print(f"\n{'='*50}")
print(f"VT Key loaded : {'YES' if key else 'NO'} ({len(key)} chars)")
print(f"{'='*50}\n")

# Test 1: VirusTotal
print("[1] VirusTotal - checking https://google.com")
url = "https://google.com"
url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
try:
    r = requests.get(
        f"https://www.virustotal.com/api/v3/urls/{url_id}",
        headers={"x-apikey": key},
        timeout=10
    )
    print(f"    HTTP {r.status_code}")
    if r.status_code == 200:
        stats = r.json().get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
        total = sum(stats.values())
        print(f"    SUCCESS - {stats.get('malicious', 0)}/{total} engines flagged")
    elif r.status_code == 404:
        print("    KEY VALID - URL not in cache yet (normal)")
    elif r.status_code == 401:
        print("    INVALID KEY")
    else:
        print(f"    HTTP {r.status_code}: {r.text[:100]}")
except Exception as e:
    print(f"    ERROR: {e}")

# Test 2: URLhaus
print("\n[2] URLhaus - checking https://google.com")
try:
    r = requests.post("https://urlhaus-api.abuse.ch/v1/url/", data={"url": "https://google.com"}, timeout=8)
    print(f"    HTTP {r.status_code}")
    print(f"    SUCCESS - query_status: {r.json().get('query_status', '?')}")
except Exception as e:
    print(f"    ERROR: {e}")

# Test 3: PhishTank
print("\n[3] PhishTank - checking https://google.com")
try:
    r = requests.post("https://checkurl.phishtank.com/checkurl/", data={"url": "https://google.com", "format": "json"}, timeout=8)
    print(f"    HTTP {r.status_code}")
    print(f"    SUCCESS - in_database: {r.json().get('results', {}).get('in_database', '?')}")
except Exception as e:
    print(f"    ERROR: {e}")

print("\nAll done.")
