import os
import sys
import pytest

# Ensure root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.bec_detector import predict_bec_intent


def test_bec_ceo_wire_fraud_detection():
    """Verify detection of zero-link CEO wire transfer fraud."""
    email_text = "I am in an urgent meeting with clients and cannot talk right now. Please process an immediate wire transfer of $15,000 to our vendor bank account today."
    res = predict_bec_intent(email_text, urls_found=0)

    assert "is_bec" in res
    assert res["is_bec"] is True
    assert res["bec_score"] >= 60
    assert "CEO Wire Fraud" in res["intent_category"]
    assert len(res["signals"]) > 0


def test_bec_gift_card_scam_detection():
    """Verify detection of zero-link gift card scam requests."""
    email_text = "Are you at your desk? I need you to purchase 5 Apple gift cards for a client presentation. Send the codes to me ASAP."
    res = predict_bec_intent(email_text, urls_found=0)

    assert res["is_bec"] is True
    assert res["bec_score"] >= 45
    assert "Gift Card Scam" in res["intent_category"] or "Executive Coercion" in res["intent_category"]


def test_bec_clean_email():
    """Verify normal business email is classified as Clean."""
    email_text = "Hi Sarah, please find attached the Q3 project update notes for our team review tomorrow morning. Best regards."
    res = predict_bec_intent(email_text, urls_found=0)

    assert res["is_bec"] is False
    assert res["bec_score"] < 40
    assert res["intent_category"] == "Clean"
