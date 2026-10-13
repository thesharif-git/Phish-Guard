import os
import sys
import pytest
import numpy as np
import cv2

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vision_scanner import (
    decode_image_bytes,
    detect_qr_codes,
    preprocess_image_for_ocr,
    analyze_image_attachment
)


def test_opencv_version():
    """Verify OpenCV 5.x is active."""
    assert cv2.__version__.startswith("5.")


def test_decode_image_bytes():
    """Verify decoding raw synthetic image bytes."""
    # Create a synthetic 100x100 RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (255, 0, 0)
    _, encoded = cv2.imencode(".png", img)
    raw_bytes = encoded.tobytes()

    decoded = decode_image_bytes(raw_bytes)
    assert decoded is not None
    assert decoded.shape == (100, 100, 3)


def test_preprocess_image_for_ocr():
    """Verify preprocessing pipeline converts image to binarized array."""
    img = np.ones((200, 200, 3), dtype=np.uint8) * 180
    cv2.putText(img, "PHISH TEST", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    processed = preprocess_image_for_ocr(img)
    assert processed is not None
    assert len(processed.shape) == 2  # Grayscale / binarized


def test_analyze_image_attachment():
    """Verify image attachment analyzer returns correct schema."""
    img = np.zeros((150, 150, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".jpg", img)
    raw_bytes = encoded.tobytes()

    res = analyze_image_attachment(raw_bytes, filename="invoice.jpg")
    assert res["is_valid_image"] is True
    assert res["width"] == 150
    assert res["height"] == 150
    assert res["filename"] == "invoice.jpg"


def test_detect_qr_code_quishing():
    """Verify real QR code generation, attachment analysis and URL decoding."""
    target_url = "https://phish-portal-alert.xyz/auth/login"
    encoder = cv2.QRCodeEncoder.create()
    qr_matrix = encoder.encode(target_url)

    # Scale up QR matrix to standard readable size (e.g. 200x200)
    qr_img = cv2.resize(qr_matrix, (200, 200), interpolation=cv2.INTER_NEAREST)
    qr_bgr = cv2.cvtColor(qr_img, cv2.COLOR_GRAY2BGR)

    _, encoded = cv2.imencode(".png", qr_bgr)
    raw_bytes = encoded.tobytes()

    res = analyze_image_attachment(raw_bytes, filename="qr_flyer.png")
    assert res["is_valid_image"] is True
    assert res["has_qr"] is True
    assert len(res["qr_urls"]) > 0
    assert target_url in res["qr_urls"]
