import re
import ssl
import socket
import requests
from bs4 import BeautifulSoup
from core.logger import logger
from urllib.parse import urlparse

# Realistic User-Agent to avoid immediate bot-blocking
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

BRAND_TARGETS = {
    "paypal": ["paypal.com", "paypal.co.uk"],
    "microsoft": ["microsoft.com", "live.com", "office.com"],
    "apple": ["apple.com", "icloud.com"],
    "google": ["google.com", "gmail.com"],
    "amazon": ["amazon.com", "aws.amazon.com"],
    "netflix": ["netflix.com"],
    "facebook": ["facebook.com"],
    "chase": ["chase.com"],
    "bank of america": ["bankofamerica.com"]
}

# ── SSL Certificate Inspector ──────────────────────────────────────────────────
def get_ssl_info(hostname: str, port: int = 443) -> dict:
    """
    Retrieve the real SSL certificate details for a hostname.
    Returns issuer, subject, expiry date and validity flag.
    """
    result = {
        "available": False,
        "valid": False,
        "issuer": "Unknown",
        "subject": "Unknown",
        "expires": "Unknown",
        "error": None,
    }
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((hostname, port), timeout=3),
            server_hostname=hostname
        ) as conn:
            cert = conn.getpeercert()

        # Extract issuer common name / organisation
        issuer_fields = {k: v for tup in cert.get("issuer", ()) for k, v in tup}
        issuer_org = issuer_fields.get("organizationName", "") or issuer_fields.get("commonName", "Unknown")

        # Subject CN
        subject_fields = {k: v for tup in cert.get("subject", ()) for k, v in tup}
        subject_cn = subject_fields.get("commonName", hostname)

        # Expiry — format is "May 12 00:00:00 2026 GMT"
        expires_raw = cert.get("notAfter", "")
        import datetime
        try:
            expires_dt = datetime.datetime.strptime(expires_raw, "%b %d %H:%M:%S %Y %Z")
            expires_str = expires_dt.strftime("%Y-%m-%d")
            is_valid = expires_dt > datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        except Exception:
            expires_str = expires_raw
            is_valid = True  # cert was validated by ssl module, so assume valid

        result.update({
            "available": True,
            "valid": is_valid,
            "issuer": issuer_org,
            "subject": subject_cn,
            "expires": expires_str,
        })
    except ssl.SSLCertVerificationError:
        result["valid"] = False
        result["issuer"] = "Invalid / Self-Signed"
        result["available"] = True
        result["error"] = "Certificate verification failed"
    except (socket.timeout, ConnectionRefusedError, OSError) as e:
        result["error"] = f"Connection failed: {type(e).__name__}"
    except Exception as e:
        result["error"] = str(e)[:80]
    return result


# ── CMS / Tech fingerprint patterns ───────────────────────────────────────────
_CMS_PATTERNS = [
    # WordPress
    (re.compile(r'/wp-content/|/wp-includes/|wp-json', re.I), "WordPress"),
    # Shopify
    (re.compile(r'cdn\.shopify\.com|Shopify\.theme', re.I), "Shopify"),
    # Joomla
    (re.compile(r'/components/com_|joomla', re.I), "Joomla"),
    # Drupal
    (re.compile(r'Drupal\.settings|/sites/default/files/', re.I), "Drupal"),
    # Wix
    (re.compile(r'static\.wixstatic\.com|wix\.com/dpages', re.I), "Wix"),
    # Squarespace
    (re.compile(r'squarespace\.com/universal/', re.I), "Squarespace"),
    # Magento
    (re.compile(r'Mage\.Cookies|skin/frontend/', re.I), "Magento"),
    # PrestaShop
    (re.compile(r'/modules/ps_|prestashop', re.I), "PrestaShop"),
]

_ANALYTICS_PATTERNS = [
    (re.compile(r'google-analytics\.com/analytics\.js|gtag\(|GoogleAnalyticsObject|googletagmanager\.com/gtag', re.I), "Google Analytics"),
    (re.compile(r'google-analytics\.com/g/collect|measurement_id|G-[A-Z0-9]{6,}', re.I), "Google Analytics 4"),
    (re.compile(r'plausible\.io/js/', re.I), "Plausible"),
    (re.compile(r'cdn\.segment\.com|analytics\.segment\.io', re.I), "Segment"),
    (re.compile(r'hotjar\.com/c/hotjar-|hj\(', re.I), "Hotjar"),
    (re.compile(r'mixpanel\.com/lib/', re.I), "Mixpanel"),
    (re.compile(r'clarity\.ms/tag/', re.I), "Microsoft Clarity"),
    (re.compile(r'matomo\.js|piwik\.js', re.I), "Matomo"),
    (re.compile(r'mc\.yandex\.ru/metrika|yaCounter', re.I), "Yandex Metrika"),
    (re.compile(r'cdn\.amplitude\.com|amplitude\.getInstance', re.I), "Amplitude"),
    (re.compile(r'heap\.io/config|window\.heap', re.I), "Heap Analytics"),
    (re.compile(r'static\.intercomcdn\.com|Intercom\(', re.I), "Intercom"),
    (re.compile(r'crisp\.chat/g\.js|CRISP_WEBSITE_ID', re.I), "Crisp"),
    (re.compile(r'shopify_analytics|ShopifyAnalytics', re.I), "Shopify Analytics"),
    (re.compile(r'static\.klaviyo\.com|klaviyo\.com/media', re.I), "Klaviyo"),
]

_TRACKING_PATTERNS = [
    (re.compile(r'googletagmanager\.com/gtm\.js', re.I), "Google Tag Manager"),
    (re.compile(r'connect\.facebook\.net/.*fbevents|fbq\(', re.I), "Facebook Pixel"),
    (re.compile(r'doubleclick\.net|googlesyndication\.com', re.I), "Google Doubleclick"),
    (re.compile(r'linkedin\.com/px/', re.I), "LinkedIn Insight"),
    (re.compile(r'snap\.licdn\.com', re.I), "Snap Pixel"),
    (re.compile(r'tiktok\.com/i18n/pixel', re.I), "TikTok Pixel"),
    (re.compile(r'twitter\.com/i/adsct|t\.co/i/adsct', re.I), "Twitter Pixel"),
]


def _detect_language(html_text: str, headers: dict, cookies: dict, url: str, cms_list: list) -> str:
    """
    Detect programming language from HTTP headers, cookies, URL patterns, meta tags, and CMS.
    Returns a human-readable string like 'PHP', 'Python', 'Ruby', etc.
    """
    # 1. X-Powered-By header — most reliable direct signal
    powered_by = headers.get('X-Powered-By', '') or headers.get('x-powered-by', '')
    if powered_by:
        pb = powered_by.lower()
        if 'php' in pb:           return f"PHP ({powered_by.split('/')[0].strip()})"
        if 'asp.net' in pb:       return "ASP.NET (C#)"
        if 'express' in pb:       return "Node.js (Express)"
        if 'next.js' in pb:       return "Node.js (Next.js)"
        if 'python' in pb:        return "Python"
        if 'ruby' in pb:          return "Ruby"
        if 'java' in pb:          return "Java"

    # 2. X-Generator header
    generator = headers.get('X-Generator', '') or headers.get('x-generator', '')
    if 'WordPress' in generator:  return "PHP (WordPress)"
    if 'Drupal' in generator:     return "PHP (Drupal)"

    # 3. Meta generator tag in HTML
    meta_match = re.search(r'<meta[^>]+name=["\']generator["\'][^>]+content=["\']([^"\']+)["\']', html_text, re.I)
    if meta_match:
        gen = meta_match.group(1)
        if 'WordPress' in gen:    return f"PHP (WordPress {gen.split(' ')[-1]})"
        if 'Joomla' in gen:       return "PHP (Joomla)"
        if 'Drupal' in gen:       return "PHP (Drupal)"
        if 'Wix' in gen:          return "JavaScript (Wix)"
        if 'Squarespace' in gen:  return "JavaScript (Squarespace)"

    # 4. CMS-based language inference
    if cms_list:
        cms = cms_list[0]
        if cms in ('WordPress', 'Joomla', 'Drupal', 'Magento', 'PrestaShop'):
            return "PHP"
        if cms == 'Shopify':      return "Ruby (Liquid/Shopify)"
        if cms == 'Wix':          return "JavaScript (Node.js)"
        if cms == 'Squarespace':  return "JavaScript (Node.js)"

    # 5. Cookie names — session cookie type reveals backend
    for cookie_name in cookies:
        cn = cookie_name.upper()
        if cn == 'PHPSESSID':     return "PHP"
        if cn == 'JSESSIONID':    return "Java (Servlet)"
        if cn.startswith('ASP.NET_SESSIONID') or cn == 'ASPSESSIONID':
            return "ASP.NET (C#)"
        if cn == '_RAILS_SESSION' or cn.startswith('_SESSION_'):
            return "Ruby (Rails)"
        if cn == 'DJANGO_SESSION' or cn == 'SESSIONID':
            return "Python (Django)"

    # 6. URL extension hint
    path = url.split('?')[0].lower()
    if path.endswith('.php'):     return "PHP"
    if path.endswith(('.asp', '.aspx')): return "ASP.NET (C#)"
    if path.endswith('.jsp'):     return "Java (JSP)"
    if path.endswith('.py'):      return "Python"
    if path.endswith('.rb'):      return "Ruby"

    # 7. Server header hints
    server = (headers.get('Server', '') or '').lower()
    if 'gunicorn' in server or 'uvicorn' in server: return "Python"
    if 'passenger' in server:    return "Ruby"
    if 'tomcat' in server:       return "Java"
    if 'iis' in server:          return "ASP.NET (C#)"

    return "Unknown"


def _scan_tech(html_text: str, headers: dict = None, cookies: dict = None, url: str = "") -> dict:
    """
    Fingerprint CMS, analytics, tracking, and programming language from raw HTML + HTTP headers.
    """
    headers  = headers  or {}
    cookies  = cookies  or {}

    cms_found       = []
    analytics_found = []
    tracking_found  = []

    for pattern, name in _CMS_PATTERNS:
        if pattern.search(html_text) and name not in cms_found:
            cms_found.append(name)

    for pattern, name in _ANALYTICS_PATTERNS:
        if pattern.search(html_text) and name not in analytics_found:
            analytics_found.append(name)

    for pattern, name in _TRACKING_PATTERNS:
        if pattern.search(html_text) and name not in tracking_found:
            tracking_found.append(name)

    language = _detect_language(html_text, headers, cookies, url, cms_found)

    return {
        "cms":       cms_found,
        "analytics": analytics_found,
        "tracking":  tracking_found,
        "language":  language,
    }


def analyze_live_html(url: str) -> dict:
    """
    Downloads and analyzes the live HTML of a URL.
    Returns heuristic flags, threat score, SSL info, and technology fingerprints.
    """
    if not url.startswith("http"):
        url = "http://" + url

    parsed   = urlparse(url)
    hostname = parsed.netloc.lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]
    domain = hostname  # alias for clarity

    result = {
        "success":           False,
        "score":             0,
        "flags":             [],
        "has_password_field": False,
        "brand_mismatch":    False,
        "obfuscated_js":     False,
        "server":            "Unknown",
        "error":             None,
        # Tech fingerprints
        "ssl_info":    {},
        "tech":        {"cms": [], "analytics": [], "tracking": []},
    }

    # ── SSL Certificate ──────────────────────────────────────────────────
    if url.startswith("https://"):
        result["ssl_info"] = get_ssl_info(hostname)
    else:
        result["ssl_info"] = {"available": False, "valid": False, "issuer": "No HTTPS", "error": "Not HTTPS"}

    try:
        # 1. Fetch HTML with Primary Headers
        resp = None
        html = ""
        server = "Unknown"
        headers_dict = {}
        cookies_dict = {}
        status_code = None

        try:
            resp = requests.get(url, headers=HEADERS, timeout=4, allow_redirects=True)
            status_code = resp.status_code
        except Exception:
            pass

        # Fallback 1: If 403 or failed, try acting like cURL (bypasses some WAFs)
        if not resp or status_code in (403, 401, 406):
            try:
                curl_headers = {"User-Agent": "curl/7.88.1", "Accept": "*/*"}
                resp = requests.get(url, headers=curl_headers, timeout=4, allow_redirects=True)
                status_code = resp.status_code
            except Exception:
                pass

        # Fallback 2: Urllib request (bypasses requests-specific blocking)
        if not resp or status_code in (403, 401, 406):
            import urllib.request
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=4) as response:
                    html = response.read().decode('utf-8', errors='ignore')
                    server = response.headers.get("Server", "Unknown")
                    headers_dict = dict(response.headers)
                    status_code = response.getcode()
            except Exception as e:
                result["error"] = "All connection attempts blocked by target server"
                return result

        if resp and status_code:
            if status_code != 200 and not html:
                result["error"] = f"HTTP {status_code}"
                return result
            if not html:
                html = resp.text
                server = resp.headers.get("Server", "Unknown")
                headers_dict = dict(resp.headers)
                cookies_dict = resp.cookies.get_dict()

        soup = BeautifulSoup(html, "html.parser")
        result["success"] = True
        result["server"]  = server
        score = 0

        # ── Technology fingerprinting (HTML + HTTP headers + cookies) ────
        result["tech"] = _scan_tech(html, headers=headers_dict, cookies=cookies_dict, url=url)

        # 2. Heuristic: Password field detection
        password_inputs = soup.find_all("input", type=lambda t: t and t.lower() == "password")
        if password_inputs:
            result["has_password_field"] = True
            if not url.startswith("https://"):
                result["flags"].append("Live HTML: Contains a password input field on an insecure (HTTP) page.")
                score += 85
            else:
                result["flags"].append("Live HTML: Contains a password input field.")
                score += 30

        # 3. Heuristic: Brand Mismatch in Title
        title_tag  = soup.find("title")
        title_text = title_tag.text.lower() if title_tag else ""

        for brand, official_domains in BRAND_TARGETS.items():
            if brand in title_text:
                is_official = any(domain == off or domain.endswith("." + off) for off in official_domains)
                if not is_official:
                    result["brand_mismatch"] = True
                    result["flags"].append(f"Live HTML: Page title mimics '{brand.title()}' but domain is unofficial.")
                    score += 90
                    break

        # 4. Heuristic: Obfuscated JavaScript
        scripts = soup.find_all("script")
        for script in scripts:
            if script.string:
                js = script.string.lower()
                if "eval(unescape(" in js or "document.write(unescape(" in js or "atob(" in js:
                    result["obfuscated_js"] = True
                    result["flags"].append("Live HTML: Contains obfuscated or packed JavaScript.")
                    score += 60
                    break

        # 5. Suspicious External Includes (raw IP addresses)
        external_ips = 0
        for tag in soup.find_all(['script', 'img', 'link']):
            src = tag.get('src') or tag.get('href')
            if src and re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', src):
                external_ips += 1

        if external_ips > 0:
            result["flags"].append(f"Live HTML: Loads {external_ips} resources directly from raw IP addresses.")
            score += 40

        # 6. Suspicious SSL (self-signed / expired)
        ssl_i = result["ssl_info"]
        if ssl_i.get("available") and not ssl_i.get("valid"):
            result["flags"].append(f"SSL: Certificate invalid or self-signed ({ssl_i.get('issuer', 'Unknown')})")
            score += 30

        result["score"] = min(score, 100)
        return result

    except requests.exceptions.Timeout:
        result["error"] = "Connection timed out"
    except requests.exceptions.RequestException as e:
        result["error"] = f"Connection failed: {e}"
    except Exception as e:
        logger.error(f"HTML Scanner error: {e}")
        result["error"] = "Parsing error"

    return result
