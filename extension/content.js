// Content Script for PhishGuard OriginTrace Extension

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "showScanToast" && message.data) {
    displayToast(message.data);
  }
});

// Toast notification for user-initiated right-click scans
function displayToast(data) {
  let toast = document.getElementById("phishguard-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "phishguard-toast";
    toast.style.cssText = `
      position: fixed;
      bottom: 24px;
      right: 24px;
      z-index: 9999999;
      background: #0f172a;
      color: #f8fafc;
      border: 1px solid #334155;
      border-radius: 10px;
      padding: 14px 18px;
      font-family: system-ui, -apple-system, sans-serif;
      font-size: 13px;
      box-shadow: 0 15px 35px -5px rgba(0, 0, 0, 0.6);
      transition: all 0.3s ease;
      max-width: 340px;
    `;
    document.body.appendChild(toast);
  }

  const badgeColor = data.status === "PHISHING" ? "#ef4444" : (data.status === "SUSPICIOUS" ? "#eab308" : "#22c55e");

  toast.innerHTML = `
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
      <strong style="color:#60a5fa; font-size:13px; display:flex; align-items:center; gap:6px;">
        ${data.icon} OriginTrace Scan
      </strong>
      <span style="background:${badgeColor}26; color:${badgeColor}; border:1px solid ${badgeColor}66; font-weight:700; padding:2px 8px; border-radius:4px; font-size:11px;">
        ${data.status} (${data.score}/100)
      </span>
    </div>
    <div style="color:#cbd5e1; font-size:12px; word-break:break-all; line-height:1.4;">
      ${data.url}
    </div>
  `;

  toast.style.opacity = "1";
  toast.style.transform = "translateY(0)";

  setTimeout(() => {
    if (toast) {
      toast.style.opacity = "0";
      toast.style.transform = "translateY(10px)";
      setTimeout(() => toast.remove(), 400);
    }
  }, 5000);
}
