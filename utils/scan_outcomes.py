"""Count saved decisions without turning review or policy into confirmed attacks."""


def scan_outcome(result):
    if not isinstance(result, dict):
        return 'unavailable'
    status = result.get('status')
    action = result.get('security_action')
    if action == 'BLOCK_BY_POLICY' or result.get('decision_basis') == 'manual_sender_block':
        return 'policy_matches'
    if status not in ('PHISHING', 'SUSPICIOUS', 'CLEAN'):
        return 'unavailable'
    if action == 'WARNING_BANNER' or status == 'SUSPICIOUS':
        return 'review_required'
    if status == 'PHISHING':
        return 'phishing'
    return 'low_risk'


def count_scan_outcomes(results):
    counts = {'phishing': 0, 'review_required': 0, 'unavailable': 0,
              'policy_matches': 0, 'low_risk': 0, 'quarantine_recommended': 0}
    for result in results:
        outcome = scan_outcome(result)
        counts[outcome] += 1
        if outcome == 'phishing' and result.get('security_action') == 'AUTO_QUARANTINE':
            counts['quarantine_recommended'] += 1
    return counts
