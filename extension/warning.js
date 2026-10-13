// OriginTrace Threat Intercept Controller

document.addEventListener("DOMContentLoaded", () => {
  const params = new URLSearchParams(window.location.search);
  const targetUrl = params.get("url") || "Unknown Destination";
  const riskScore = parseInt(params.get("score") || "85", 10);
  const reasonsRaw = params.get("reasons") || "[]";
  const targetBrand = params.get("brand") || "Financial/Enterprise Portal";

  // Elements
  const urlEl = document.getElementById("target-url");
  const scoreBadge = document.getElementById("risk-score-badge");
  const provEl = document.getElementById("pillar-provenance");
  const infraEl = document.getElementById("pillar-infrastructure");
  const deceptEl = document.getElementById("pillar-deception");
  const evidEl = document.getElementById("pillar-evidence");
  const closeBtn = document.getElementById("btn-close-tab");
  const proceedBtn = document.getElementById("btn-proceed-unsafe");

  // Display URL
  if (urlEl) urlEl.textContent = targetUrl;

  // Display Risk Score
  if (scoreBadge) {
    scoreBadge.textContent = `${riskScore}/100 HIGH RISK`;
    if (riskScore < 50) {
      scoreBadge.style.borderColor = "#f59e0b";
      scoreBadge.style.color = "#fef08a";
      scoreBadge.textContent = `${riskScore}/100 SUSPICIOUS`;
    }
  }

  // Populate Pillars
  let parsedReasons = [];
  try {
    parsedReasons = JSON.parse(reasonsRaw);
  } catch (e) {
    if (reasonsRaw) parsedReasons = [reasonsRaw];
  }

  const reasonsText = parsedReasons.length > 0 ? parsedReasons.join("; ") : "Detected fraudulent lookalike patterns.";

  if (provEl) {
    provEl.textContent = `Unauthorized impersonation of ${targetBrand}. Failed identity integrity and domain reputation checks.`;
  }

  if (infraEl) {
    infraEl.textContent = reasonsText.includes("TLD") 
      ? "Abuse-prone top-level domain and disposable hosting infrastructure detected."
      : "Hosting origin inconsistent with legitimate enterprise infrastructure.";
  }

  if (deceptEl) {
    deceptEl.textContent = reasonsText.includes("credential") || reasonsText.includes("login")
      ? "Deceptive credential harvesting triggers and deceptive URI syntax identified."
      : "Manipulative lexical indicators engineered to deceive end-users.";
  }

  if (evidEl) {
    evidEl.textContent = `Dempster-Shafer mathematical consensus: ${riskScore}% confidence in malicious verdict. Zero-trust barrier enforced.`;
  }

  // Button Actions
  if (closeBtn) {
    closeBtn.addEventListener("click", () => {
      // Close tab or go to safety
      if (chrome.tabs && chrome.tabs.getCurrent) {
        chrome.tabs.getCurrent((tab) => {
          if (tab && tab.id) {
            chrome.tabs.remove(tab.id);
          } else {
            window.location.href = "https://www.google.com";
          }
        });
      } else {
        window.close();
        window.location.href = "https://www.google.com";
      }
    });
  }

  if (proceedBtn) {
    proceedBtn.addEventListener("click", () => {
      const confirmed = confirm(
        "CRITICAL WARNING:\n\nThis website was confirmed malicious by OriginTrace. Continuing may compromise your credentials, financial data, or download malicious software.\n\nAre you absolutely sure you want to proceed?"
      );
      if (confirmed) {
        // Whitelist for 1 hour so user isn't re-intercepted
        try {
          const domain = new URL(targetUrl).hostname;
          chrome.runtime.sendMessage(
            { action: "whitelistDomainForSession", domain: domain },
            () => {
              window.location.href = targetUrl;
            }
          );
        } catch (e) {
          window.location.href = targetUrl;
        }
      }
    });
  }
});
