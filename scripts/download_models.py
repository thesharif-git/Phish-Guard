"""
Optional Model Downloader for OriginTrace / PhishGuard Local Stack
Downloads the ultra-lean Qwen2.5-0.5B-Instruct GGUF file (~390 MB)
for ~550 MB RAM execution via llama.cpp.
"""

import os
import sys
import requests

GGUF_URL = "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TARGET_FILE = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct.gguf")


def download_qwen_gguf():
    os.makedirs(MODELS_DIR, exist_ok=True)
    if os.path.exists(TARGET_FILE):
        size_mb = os.path.getsize(TARGET_FILE) / (1024 * 1024)
        print(f"[OK] Qwen2.5-0.5B GGUF already exists at: {TARGET_FILE} ({size_mb:.1f} MB)")
        return

    print(f"Downloading Qwen2.5-0.5B-Instruct (Q4_K_M ~390 MB) to {TARGET_FILE}...")
    try:
        response = requests.get(GGUF_URL, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1MB chunks

        with open(TARGET_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        sys.stdout.write(f"\rProgress: {pct:.1f}% ({downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB)")
                        sys.stdout.flush()

        print("\n[SUCCESS] Qwen2.5-0.5B GGUF downloaded successfully!")
        print("Your local AI threat explanation pipeline will now run at ~550 MB RAM.")
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if os.path.exists(TARGET_FILE):
            os.remove(TARGET_FILE)


if __name__ == "__main__":
    download_qwen_gguf()
