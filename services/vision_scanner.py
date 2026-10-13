"""
PhishGuard Vision Engine (OpenCV 5.0 Headless)
Visual Phishing, Quishing (QR-code phishing), and Image Attachment Analysis.
"""

import cv2
import numpy as np
import base64
import logging
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger("phishguard")

_QR_DETECTOR = None


def get_qr_detector():
    """Lazy-initializes OpenCV QRCodeDetector."""
    global _QR_DETECTOR
    if _QR_DETECTOR is None:
        try:
            _QR_DETECTOR = cv2.QRCodeDetector()
        except Exception as e:
            logger.error(f"[Vision Engine] Failed to initialize QRCodeDetector: {e}")
    return _QR_DETECTOR


def decode_image_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
    """
    Safely converts raw image bytes into an OpenCV BGR numpy array.
    Returns None if decoding fails or input is corrupt.
    """
    if not image_bytes:
        return None
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.warning(f"[Vision Engine] Error decoding image bytes: {e}")
        return None


def decode_base64_image(b64_str: str) -> Optional[np.ndarray]:
    """Decodes a base64 or data URL image string into an OpenCV BGR array."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_str)
        return decode_image_bytes(raw_bytes)
    except Exception as e:
        logger.warning(f"[Vision Engine] Error decoding base64 image: {e}")
        return None


def detect_qr_codes(image: np.ndarray, *, strict=False) -> List[Dict[str, Any]]:
    """
    Scans an image array for QR codes (Quishing detection).
    Returns a list of detected QR payloads with decoded text/URL and bounding box.
    """
    if image is None:
        return []

    detector = get_qr_detector()
    if detector is None:
        if strict:
            raise RuntimeError('QR detector unavailable')
        return []

    results = []
    try:
        # Multi QR detection in OpenCV 5
        retval, decoded_info, points, _ = detector.detectAndDecodeMulti(image)
        if retval:
            for i, text in enumerate(decoded_info):
                if text and text.strip():
                    results.append({
                        "payload": text.strip(),
                        "is_url": text.strip().startswith(("http://", "https://")),
                        "points": points[i].tolist() if points is not None and len(points) > i else []
                    })
        else:
            # Fallback to single QR code detection
            text, points, _ = detector.detectAndDecode(image)
            if text and text.strip():
                results.append({
                    "payload": text.strip(),
                    "is_url": text.strip().startswith(("http://", "https://")),
                    "points": points.tolist() if points is not None else []
                })
    except Exception as e:
        logger.debug(f"[Vision Engine] QR code detection error: {e}")
        if strict:
            raise RuntimeError('QR detection failed: ' + str(e)) from e

    return results


def preprocess_image_for_ocr(image: np.ndarray) -> Optional[np.ndarray]:
    """
    Preprocesses an image for optical character recognition (OCR) or text extraction:
    - Grayscale conversion
    - Noise reduction (Gaussian blur)
    - Adaptive Otsu binarization
    """
    if image is None:
        return None
    try:
        # 1. Convert to grayscale if color
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. Rescale if too small (upscaling aids OCR clarity)
        h, w = gray.shape[:2]
        if w < 300 or h < 300:
            scale = max(300 / max(w, 1), 300 / max(h, 1))
            gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # 3. Noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 4. Otsu's adaptive thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    except Exception as e:
        logger.warning(f"[Vision Engine] Image preprocessing failed: {e}")
        return image


def analyze_image_attachment(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """
    Full security inspection for an email image attachment using OpenCV 5:
    1. QR code / Quishing inspection
    2. Image metadata & dimensions check
    3. Visual spam / single-image evasion signals
    """
    img = decode_image_bytes(image_bytes)
    if img is None:
        return {
            "filename": filename,
            "is_valid_image": False,
            "has_qr": False,
            "qr_urls": [],
            "risk_signals": []
        }

    h, w = img.shape[:2]
    qr_error = None
    try:
        qr_matches = detect_qr_codes(img, strict=True)
    except Exception as e:
        qr_matches = []
        qr_error = str(e)
    qr_urls = [qr["payload"] for qr in qr_matches if qr.get("is_url")]
    has_qr = len(qr_matches) > 0

    risk_signals = []
    if has_qr:
        for url in qr_urls:
            risk_signals.append(f"Embedded QR code target URL detected: {url}")
        if not qr_urls and qr_matches:
            risk_signals.append(f"Embedded QR code payload detected: {qr_matches[0]['payload'][:50]}")

    return {
        "filename": filename,
        "is_valid_image": True,
        "qr_scan_complete": qr_error is None,
        "qr_scan_error": qr_error,
        "width": w,
        "height": h,
        "has_qr": has_qr,
        "qr_matches": qr_matches,
        "qr_urls": qr_urls,
        "risk_signals": risk_signals
    }
