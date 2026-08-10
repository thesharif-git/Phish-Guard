import os
import re

STANDARD_HEADER = """    <header class="nav" id="main-nav">
        <div class="nav-inner">
            <a href="/" class="nav-logo">
                <div class="nav-logo-icon">
                    <svg width="15" height="15" fill="none" stroke="#fff" stroke-width="2.4" viewBox="0 0 24 24">
                        <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2-1 4-3 7-3s5 2 7 3a1 1 0 0 1 1 1v7z"/>
                        <path d="m9 12 2 2 4-4" stroke-linecap="round" stroke-linejoin="round"/>
                    </svg>
                </div>
                PhishGuard
            </a>
            <nav aria-label="Primary">
                <ul class="nav-links">
                    <li><a href="/#features">Features</a></li>
                    <li><a href="/#how-it-works">How it works</a></li>
                    <li><a href="/#pricing">Pricing</a></li>
                </ul>
            </nav>
            <div class="nav-cta">
                <a href="/login" class="btn btn-ghost btn-sm">Log in</a>
                <a href="/dashboard" class="btn btn-violet btn-sm">Start free &rarr;</a>
            </div>
        </div>
    </header>"""

STANDARD_FOOTER = """    <footer>
        <div class="container">
            <div class="footer-grid">
                <div>
                    <div class="footer-brand-logo">
                        <div class="footer-brand-icon">
                            <svg width="14" height="14" fill="none" stroke="#fff" stroke-width="2.4" viewBox="0 0 24 24">
                                <path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2-1 4-3 7-3s5 2 7 3a1 1 0 0 1 1 1v7z"/>
                                <path d="m9 12 2 2 4-4" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </div>
                        PhishGuard
                    </div>
                    <p class="footer-desc">Protecting organizations from advanced phishing and email threats.</p>
                </div>
                <div>
                    <div class="footer-head">Company</div>
                    <div class="footer-links">
                        <a href="/about" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                            About
                        </a>
                        <a href="/blog" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M19 20H5a2 2 0 01-2-2V6a2 2 0 012-2h10a2 2 0 012 2v1m2 13a2 2 0 01-2-2V7m2 13a2 2 0 002-2V9a2 2 0 00-2-2h-2m-4-3H9M7 16h6M7 8h6v4H7V8z" /></svg>
                            Blog
                        </a>
                        <a href="/contact" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" /></svg>
                            Contact
                        </a>
                    </div>
                </div>
                <div>
                    <div class="footer-head">Legal</div>
                    <div class="footer-links">
                        <a href="/privacy" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"/></svg>
                            Privacy Policy
                        </a>
                        <a href="/terms" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>
                            Terms of Service
                        </a>
                        <a href="/security" class="footer-link">
                            <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/></svg>
                            Security
                        </a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <span>&copy; 2026 PhishGuard. All rights reserved.</span>
                <span>Powered by VirusTotal &middot; PhishTank &middot; URLhaus &middot; WHOIS</span>
            </div>
        </div>
    </footer>"""

def standardize_file(path):
    print(f"Standardizing: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='windows-1252') as f:
            content = f.read()

    # Replace header block
    content = re.sub(r'<!-- NAV -->\n    <header class="nav".*?</header>', STANDARD_HEADER, content, flags=re.DOTALL)
    # If comment is missing
    content = re.sub(r'<header class="nav" id="nav">.*?</header>', STANDARD_HEADER, content, flags=re.DOTALL)
    
    # Replace footer block
    content = re.sub(r'<!-- FOOTER -->\n    <footer>.*?</footer>', STANDARD_FOOTER, content, flags=re.DOTALL)
    # If comment is missing
    content = re.sub(r'<footer>.*?</footer>', STANDARD_FOOTER, content, flags=re.DOTALL)

    # Ensure stylesheet versioning
    content = content.replace('landing.css?v=12', 'landing.css?v=13')

    # Fix character corruption
    replacements = {
        'âœ“': '&check;',
        'â€¦': '&hellip;',
        'Â·': '&middot;',
        'Â©': '&copy;',
        '&bullet;': '&middot;', # Standardize bullets
    }
    for old, new in replacements.items():
        content = content.replace(old, new)

    with open(path, 'w', encoding='utf-8', newline='') as f:
        f.write(content)

if __name__ == "__main__":
    info_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates', 'info')
    for filename in os.listdir(info_dir):
        if filename.endswith(".html"):
            standardize_file(os.path.join(info_dir, filename))
    print("Verification complete.")
