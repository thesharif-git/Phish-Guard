from utils.url_scanner import run_url_heuristics


def test_brand_security_domain_is_flagged_without_flagging_official_domain():
    suspicious = run_url_heuristics("https://paypal-secure.net")
    official = run_url_heuristics("https://www.paypal.com/login")
    path_mention = run_url_heuristics("https://example.net/paypal-login")

    assert suspicious["heuristic_score"] >= 75
    assert any("paired with 'secure'" in flag for flag in suspicious["flags"])
    assert official["heuristic_score"] == 0
    assert path_mention["heuristic_score"] == 0
