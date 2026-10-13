"""Read-only Windows process snapshots for the existing model artifacts.

No fits, inference, downloads, calibration, application imports or promotion.
Loaded-process snapshots are not training peaks or whole-server memory.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.venv/Lib/site-packages'))
sys.path.insert(0, str(ROOT))
import ctypes
from ctypes import wintypes
import hashlib
import json
import time


class MemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
        (field, ctypes.c_size_t) for field in ('PeakWorkingSetSize', 'WorkingSetSize',
        'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
        'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage', 'PrivateUsage')]


def memory():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return {'working_set_mib': counters.WorkingSetSize / 2**20,
            'private_committed_mib': counters.PrivateUsage / 2**20,
            'peak_working_set_mib_so_far': counters.PeakWorkingSetSize / 2**20}


def main():
    output = ROOT / 'benchmarks/lightweight_resource_review_20261004'
    output.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'One isolated Windows process, existing artifacts loaded; no inference/training peaks, no new-model accuracy comparison',
              'memory_units': 'MiB (2**20 bytes)', 'snapshots': [{'stage': 'python_start', **memory()}], 'artifacts': []}
    started = time.perf_counter()
    import joblib
    import sklearn
    import scipy
    import lightgbm
    report['snapshots'].append({'stage': 'scientific_libraries_imported', 'elapsed_seconds': time.perf_counter()-started, **memory()})
    retained = []
    for name in ('phish_model.pkl', 'vectorizer.pkl', 'url_model.pkl', 'domain_model.pkl'):
        path = ROOT / 'models' / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        started = time.perf_counter()
        artifact = joblib.load(path)
        retained.append(artifact)
        elapsed = time.perf_counter()-started
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        entry = {'name': name, 'disk_bytes': path.stat().st_size, 'sha256': digest,
                 'fold_estimators': len(getattr(artifact, 'calibrated_classifiers_', [])), 'load_seconds': elapsed}
        report['artifacts'].append(entry)
        report['snapshots'].append({'stage': 'loaded_' + name, **memory()})
    report['artifacts_modified'] = False
    report['accuracy_claim'] = None
    (output / 'current_model_resources.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
