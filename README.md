# 🛡️ PhishGuard — AI Phishing Detection Platform

> A premium AI-powered cybersecurity tool for detecting phishing emails, scanning malicious URLs, and auditing domain reputation in real time.

![PhishGuard Banner](https://img.shields.io/badge/PhishGuard-AI%20Security-10b981?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## ✨ Features

- **🔍 Email Scanner** — Paste raw email headers/body and get an instant AI-powered phishing verdict with confidence score, risk breakdown, and key findings.
- **🔗 URL Scanner** — Analyze URLs for phishing, malware, suspicious redirects, SSL issues, and domain age.
- **🌐 Domain Reputation Auditor** — Deep-dive WHOIS, VirusTotal, PhishTank, URLhaus, SSL cert, and geolocation data for any domain.
- **📊 Security Dashboard** — Real-time command center showing threat metrics, scan history, and posture summary.
- **📁 Scan History** — Filterable, searchable log of all past scans.
- **⚙️ Settings** — User profile, API key management, notification preferences, and appearance.
- **🧩 Browser Extension** — Lightweight Chrome/Edge extension for inline URL scanning.

---

## 🎨 Design

PhishGuard uses a **Premium Obsidian Dark Theme**:
- Deep Obsidian Charcoal background (`#09090b`)
- Elevated card surfaces (`#121215`, `#1a1a20`)
- **Cyber Emerald** accent (`#10b981`) — zero blue, zero navy
- Smooth glassmorphism, micro-animations, and crisp typography (Plus Jakarta Sans / Inter)

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python · FastAPI · SQLAlchemy · Alembic |
| Frontend | Jinja2 Templates · Tailwind CSS (CDN) · Vanilla JS |
| AI/ML | Scikit-learn phishing classifier |
| Threat Intel | VirusTotal API · PhishTank · URLhaus · WHOIS |
| Auth | JWT / session-based |
| Database | SQLite (dev) / PostgreSQL (prod) |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/phishguard.git
cd phishguard
```

### 2. Create a virtual environment
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# or
source .venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 5. Run database migrations
```bash
alembic upgrade head
```

### 6. Start the server
```bash
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000` 🚀

---

## 🔑 Environment Variables

See [`.env.example`](.env.example) for a full list. Key variables:

| Variable | Description |
|---|---|
| `SECRET_KEY` | JWT signing secret |
| `VIRUSTOTAL_API_KEY` | VirusTotal v3 API key |
| `DATABASE_URL` | SQLite or PostgreSQL URL |

---

## 📁 Project Structure

```
phishguard/
├── main.py              # FastAPI app entry point
├── routers/             # API & page route handlers
├── services/            # Business logic (scanner, threat intel)
├── models/              # SQLAlchemy ORM models
├── templates/           # Jinja2 HTML templates
├── static/              # CSS, JS, images
├── core/                # Config, security, auth utilities
├── alembic/             # DB migration scripts
├── training/            # ML model training scripts
└── extension/           # Browser extension source
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

*Built with ❤️ and Cyber Emerald.*
