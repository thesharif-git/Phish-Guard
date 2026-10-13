// PhishGuard Threat Scanner & OriginTrace Auto-Interceptor
// Background Service Worker (Manifest V3)

const BACKEND_BASE = "http://localhost:8000";

const SAFE_DOMAINS = new Set([
  "google.com", "www.google.com", "youtube.com", "www.youtube.com",
  "github.com", "microsoft.com", "apple.com", "wikipedia.org",
  "amazon.com", "cloudflare.com", "linkedin.com", "x.com", "twitter.com",
  "facebook.com", "instagram.com", "reddit.com", "stackoverflow.com",
  "localhost", "127.0.0.1"
]);

// Temporary session bypass for users who clicked "Proceed Anyway"
const sessionBypassedDomains = new Set();

// In-memory cache for fast lookups (url -> result)
const urlCache = new Map();

// Context Menu Setup
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "phishguard-scan-link",
    title: "🛡️ Scan link with PhishGuard AI",
    contexts: ["link"]
  });
  console.log("[PhishGuard] Extension installed & active.");
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "phishguard-scan-link" && info.linkUrl) {
    scanLinkUrl(info.linkUrl, tab);
  }
});

// Message listener for whitelist requests & quick checks
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "whitelistDomainForSession" && message.domain) {
    sessionBypassedDomains.add(message.domain.toLowerCase());
    sendResponse({ success: true });
    return true;
  }
  if (message.action === "quickCheckUrl" && message.url) {
    checkUrlAgainstOriginTrace(message.url).then(res => sendResponse(res));
    return true;
  }
});

// Fast URL evaluation against OriginTrace engine
async function checkUrlAgainstOriginTrace(url) {
  try {
    const urlObj = new URL(url);
    const domain = urlObj.hostname.toLowerCase().replace(/^www\./, "");

    // Check safe list
    for (const safe of SAFE_DOMAINS) {
      if (domain === safe || domain.endsWith("." + safe)) {
        return { status: "CLEAN", risk_score: 0, safe: true };
      }
    }

    // Check session bypass
    if (sessionBypassedDomains.has(domain)) {
      return { status: "CLEAN", risk_score: 0, bypassed: true };
    }

    // Check cache
    if (urlCache.has(url)) {
      return urlCache.get(url);
    }

    const response = await fetch(`${BACKEND_BASE}/api/extension/quick-check`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });

    if (response.ok) {
      const data = await response.json();
      urlCache.set(url, data);
      return data;
    }
  } catch (err) {
    // If backend is unreachable or offline, fail open to avoid breaking browsing
    console.debug("[PhishGuard] Backend check skipped (offline/error):", err);
  }
  return { status: "CLEAN", risk_score: 0 };
}

// ── Autonomous Navigation Interception ────────────────────────────────────────
chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Only intercept the top-level main frame (not iframes/ads)
  if (details.frameId !== 0) return;

  const url = details.url;
  if (!url || !url.startsWith("http")) return;

  try {
    const urlObj = new URL(url);
    const domain = urlObj.hostname.toLowerCase();

    // Skip localhost and safe authority domains
    if (domain === "localhost" || domain === "127.0.0.1") return;
    for (const safe of SAFE_DOMAINS) {
      if (domain === safe || domain.endsWith("." + safe)) return;
    }

    if (sessionBypassedDomains.has(domain)) return;

    // Check threat
    const threat = await checkUrlAgainstOriginTrace(url);

    if (threat && (threat.status === "PHISHING" || threat.risk_score >= 70)) {
      console.warn(`[PhishGuard OriginTrace] Intercepted Phishing Link: ${url} (Score: ${threat.risk_score})`);

      const warningUrl = chrome.runtime.getURL("warning.html") +
        `?url=${encodeURIComponent(url)}` +
        `&score=${threat.risk_score || 90}` +
        `&status=${threat.status || 'PHISHING'}` +
        `&reasons=${encodeURIComponent(JSON.stringify(threat.reasons || []))}` +
        `&brand=${encodeURIComponent(threat.origintrace_pillars?.pillar1_provenance?.target_brand || 'Protected Brand')}`;

      // Redirect the tab immediately to the OriginTrace Warning Screen
      chrome.tabs.update(details.tabId, { url: warningUrl });
    }
  } catch (e) {
    console.debug("[PhishGuard] Intercept parse error:", e);
  }
});

// Manual Scan from Right Click Context Menu
async function scanLinkUrl(linkUrl, tab) {
  try {
    const data = await checkUrlAgainstOriginTrace(linkUrl);
    const status = data.status || "CLEAN";
    const score = data.risk_score || 0;

    let icon = "🟢";
    if (status === "PHISHING" || score >= 70) icon = "🚨";
    else if (status === "SUSPICIOUS" || score >= 35) icon = "⚠️";

    if (tab && tab.id) {
      chrome.tabs.sendMessage(tab.id, {
        action: "showScanToast",
        data: {
          url: linkUrl,
          status: status,
          score: score,
          icon: icon
        }
      });
    }
  } catch (err) {
    console.error("[PhishGuard Extension] Scan error:", err);
  }
}
