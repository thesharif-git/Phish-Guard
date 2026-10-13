document.addEventListener('DOMContentLoaded', () => {
  const scanBtn = document.getElementById('scanBtn');
  const scanCurrentBtn = document.getElementById('scanCurrentBtn');
  const urlInput = document.getElementById('urlInput');
  const loading = document.getElementById('loading');
  const resultCard = document.getElementById('resultCard');
  const verdictBadge = document.getElementById('verdictBadge');
  const scoreMeter = document.getElementById('scoreMeter');
  const domainDetail = document.getElementById('domainDetail');
  const reasonDetail = document.getElementById('reasonDetail');

  const API_ENDPOINT = 'http://localhost:8000/api/scan-url';

  // Automatically fetch active tab URL for convenience
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs && tabs[0] && tabs[0].url) {
      const activeUrl = tabs[0].url;
      if (activeUrl.startsWith('http://') || activeUrl.startsWith('https://')) {
        scanCurrentBtn.textContent = `📍 Scan: ${activeUrl.substring(0, 35)}...`;
        scanCurrentBtn.dataset.url = activeUrl;
      } else {
        scanCurrentBtn.style.display = 'none';
      }
    }
  });

  scanCurrentBtn.addEventListener('click', () => {
    const url = scanCurrentBtn.dataset.url;
    if (url) {
      urlInput.value = url;
      runScan(url);
    }
  });

  scanBtn.addEventListener('click', () => {
    const url = urlInput.value.trim();
    if (url) {
      runScan(url);
    }
  });

  async function runScan(targetUrl) {
    loading.style.display = 'block';
    resultCard.style.display = 'none';

    try {
      const response = await fetch(API_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: targetUrl })
      });

      if (!response.ok) {
        throw new Error(`Server returned status ${response.status}`);
      }

      const data = await response.json();
      displayResult(data);
    } catch (err) {
      loading.style.display = 'none';
      resultCard.style.display = 'block';
      verdictBadge.textContent = 'ERROR';
      verdictBadge.className = 'verdict-badge badge-phishing';
      scoreMeter.textContent = '!';
      domainDetail.textContent = 'Connection Error';
      reasonDetail.textContent = 'Could not connect to PhishGuard backend server at http://localhost:8000.';
    }
  }

  function displayResult(data) {
    loading.style.display = 'none';
    resultCard.style.display = 'block';

    const score = data.final_score || 0;
    const status = (data.status || 'CLEAN').toUpperCase();

    scoreMeter.textContent = `${score}/100`;

    if (status === 'PHISHING' || score >= 70) {
      verdictBadge.textContent = 'PHISHING';
      verdictBadge.className = 'verdict-badge badge-phishing';
    } else if (status === 'SUSPICIOUS' || score >= 35) {
      verdictBadge.textContent = 'CAUTION';
      verdictBadge.className = 'verdict-badge badge-suspicious';
    } else {
      verdictBadge.textContent = 'SAFE';
      verdictBadge.className = 'verdict-badge badge-safe';
    }

    domainDetail.textContent = `Target: ${data.url || data.domain || 'Scanned link'}`;
    reasonDetail.textContent = data.action_recommendation || data.recommendation || `Risk level: ${data.risk_level || 'normal'}`;
  }
});
