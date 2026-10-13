"""Local operation/resource smoke check. Synthetic emails do not measure accuracy."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import http.cookiejar
import json
from pathlib import Path
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class Counters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
        'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', wintypes.DWORD), ('load_percent', wintypes.DWORD)] + [
        (name, ctypes.c_ulonglong) for name in ('total_physical', 'available_physical',
        'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual',
        'available_extended_virtual')]


def system_memory():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'physical_total_mib': status.total_physical / 2**20,
            'physical_available_mib': status.available_physical / 2**20,
            'memory_load_percent': status.load_percent}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmarks/application_check_20261004')
    parser.add_argument('--preloaded', action='store_true', help='Describe first request accurately when startup preloading was enabled')
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x410, False, args.pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]

    def memory():
        c = Counters()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {'working_set_mib': c.ws / 2**20, 'private_committed_mib': c.private / 2**20,
                'peak_working_set_mib': c.peak_ws / 2**20,
                'peak_process_commit_mib': c.peak_pagefile / 2**20}

    report = {'pid': args.pid, 'scope': 'Live single-worker application; synthetic Email operation checks, not accuracy evaluation',
              'startup_preloaded': args.preloaded,
              'pages': [], 'scans': [], 'checks': [], 'snapshots': [], 'complete': False}
    artifacts = [ROOT / 'models' / n for n in ('phish_model.pkl', 'vectorizer.pkl', 'url_model.pkl', 'domain_model.pkl', 'conformal_params.json')]
    initial_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
    report['snapshots'].append({'stage': 'startup_idle', **memory()})
    report['system_memory_before_requests'] = system_memory()
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    base = 'http://127.0.0.1:8000'

    def request(path, data=None, method=None, content_type=None, timeout=240):
        headers = {'Content-Type': content_type} if content_type else {}
        req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
        started = time.perf_counter()
        try:
            with client.open(req, timeout=timeout) as response:
                return response.status, response.read(), time.perf_counter() - started
        except urllib.error.HTTPError as e:
            return e.code, e.read(), time.perf_counter() - started

    def json_request(path, obj, method='POST'):
        return request(path, json.dumps(obj).encode(), method, 'application/json')

    def save():
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    for path in ('/', '/register', '/login', '/scan', '/url-scan', '/domain-reputation', '/dashboard', '/pricing', '/docs', '/security', '/blog', '/privacy', '/terms', '/api/docs', '/openapi.json'):
        status, body, seconds = request(path)
        entry = {'path': path, 'status': status, 'seconds': round(seconds, 3)}
        report['pages'].append(entry)
        print(json.dumps(entry), flush=True)
        if path in ('/', '/register', '/scan') and status == 200:
            assets = sorted(set(re.findall(r'(?:src|href)=[\"\'](/static/[^\"\']+)', body.decode(errors='replace'))))
            for asset in assets:
                asset_status, _, _ = request(asset)
                report['checks'].append({'asset': asset, 'status': asset_status})
    status, _, _ = request('/api/scan-email', b'content=Synthetic+test', 'POST', 'application/x-www-form-urlencoded')
    report['checks'].append({'check': 'unauthenticated_email_scan', 'status': status, 'expected': 401})
    email = 'resource-check-' + secrets.token_hex(6) + '@example.invalid'
    password = 'LocalCheck9!' + secrets.token_hex(16)
    status, _, _ = json_request('/auth/register', {'name': 'Local operation check', 'email': email, 'password': password})
    if status != 200:
        raise RuntimeError('Test registration failed: ' + str(status))
    status, _, _ = request('/auth/login', urllib.parse.urlencode({'username': email, 'password': password}).encode(), 'POST', 'application/x-www-form-urlencoded')
    if status != 200:
        raise RuntimeError('Test login failed: ' + str(status))
    status, _, _ = json_request('/api/profile', {'browser_alerts': False, 'weekly_reports': False}, 'PATCH')
    if status != 200:
        raise RuntimeError('Cannot disable test-account email alerts: ' + str(status))
    report['checks'].append({'check': 'test_registration_login_preferences', 'status': status})
    status, _, _ = request('/api/scan-email', b'', 'POST', 'application/x-www-form-urlencoded')
    report['checks'].append({'check': 'empty_authenticated_email_scan', 'status': status, 'expected': 400})
    messages = [
        ('first_plain_text' if args.preloaded else 'cold_plain_text', 'Synthetic meeting reminder', 'This is a synthetic local application test. The team meeting is tomorrow at ten. Please bring the project notes. No action on any account is requested.'),
        ('warm_eml_upload', 'Synthetic urgent payment request', 'Synthetic local test only. Urgently send a wire transfer today to a new bank account. Keep this confidential and bypass the usual approval process.'),
        ('unavailable_vocabulary_plain_text', 'qzxvbrqzxvbr', 'qzxvbrqzxvbr ' * 30)]
    for stage, subject, content in messages:
        samples = []
        done = threading.Event()
        def sample():
            while not done.wait(0.2):
                samples.append(memory())
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            if stage != 'warm_eml_upload':
                data = urllib.parse.urlencode({'content': content, 'sender': 'tester@example.invalid', 'subject': subject}).encode()
                status, body, seconds = request('/api/scan-email', data, 'POST', 'application/x-www-form-urlencoded')
            else:
                boundary = 'LocalSmoke' + secrets.token_hex(8)
                eml = 'From: tester@example.invalid\r\nTo: recipient@example.invalid\r\nSubject: ' + subject + '\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\n' + content
                data = ('--' + boundary + '\r\nContent-Disposition: form-data; name="file"; filename="synthetic-test.eml"\r\nContent-Type: message/rfc822\r\n\r\n' + eml + '\r\n--' + boundary + '--\r\n').encode()
                status, body, seconds = request('/api/scan-email', data, 'POST', 'multipart/form-data; boundary=' + boundary)
        finally:
            done.set()
            thread.join()
        result = json.loads(body) if status == 200 else {'error_response': body.decode(errors='replace')[:500]}
        (output / (stage + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
        entry = {'stage': stage, 'http_status': status, 'seconds': round(seconds, 3),
                 'scan_id': result.get('scan_id'), 'status': result.get('status'), 'risk_score': result.get('final_score'),
                 'peak_sampled_private_mib': max((v['private_committed_mib'] for v in samples), default=None),
                 'peak_sampled_working_set_mib': max((v['working_set_mib'] for v in samples), default=None),
                 'after': memory(), 'result_keys': sorted(result)}
        report['scans'].append(entry)
        if stage == 'unavailable_vocabulary_plain_text':
            report['checks'].append({'check': 'unavailable_email_has_no_invented_score',
                'passed': status == 200 and result.get('final_score') is None
                    and result.get('content_analysis', {}).get('ml_probability') is None
                    and result.get('security_action') == 'WARNING_BANNER'})
        report['snapshots'].append({'stage': stage + '_after', **memory()})
        print(json.dumps(entry), flush=True)
        save()
    status, body, _ = request('/api/scan-history')
    history = json.loads(body) if status == 200 else None
    (output / 'test_account_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    report['checks'].append({'check': 'authenticated_scan_history', 'status': status})
    rows = (history or {}).get('scans', [])
    report['checks'].append({'check': 'one_owned_history_row_per_scan',
                             'passed': len(rows) == len(messages) and len({r['scan_id'] for r in rows}) == len(messages)})
    report['checks'].append({'check': 'unavailable_email_history_score_is_null',
                            'passed': len([r for r in rows if r.get('subject') == 'qzxvbrqzxvbr'
                                           and r.get('risk_score') is None]) == 1})
    for path in ['/dashboard', '/inbox', '/reports', '/settings', '/campaigns'] + ['/email/' + row['scan_id'] for row in rows]:
        status, _, seconds = request(path)
        report['checks'].append({'check': 'authenticated_page', 'path': path, 'status': status, 'seconds': round(seconds, 3)})
    report['test_account'] = 'Reserved example.invalid local test account; alerts disabled; contains only the three synthetic scans.'
    report['protected_artifacts_unchanged'] = all(hashlib.sha256(p.read_bytes()).hexdigest() == initial_hashes[p.name] for p in artifacts)
    report['operational_checks_passed'] = (all(p['status'] == 200 for p in report['pages'])
        and all(s['http_status'] == 200 for s in report['scans'])
        and all(c.get('status', 200) == c.get('expected', 200) and c.get('passed', True) for c in report['checks']))
    report['complete'] = True
    report['system_memory_after_requests'] = system_memory()
    save()
    print('Report saved: ' + str(output / 'results.json'), flush=True)


if __name__ == '__main__':
    main()
