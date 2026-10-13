import sys
from types import ModuleType

from services.email_runtime import preload_email_components


def fake_components(monkeypatch, *, unavailable_bec=False):
    calls = []
    engine, vector, bec = (ModuleType('services.' + name) for name in ('scan_engine', 'vector_store', 'bec_detector'))
    engine.ML_MODEL = engine.URL_MODEL = engine.DOMAIN_MODEL = None
    def models():
        calls.append('models')
        engine.ML_MODEL = engine.URL_MODEL = engine.DOMAIN_MODEL = object()
    engine.load_ml_model = models
    def value(name, unavailable=False):
        def load():
            calls.append(name)
            return None if unavailable else object()
        return load
    vector._get_embedding_model = value('embedding')
    vector._get_chroma_collection = value('collection')
    bec._get_bec_classifier = value('bec', unavailable_bec)
    import services
    for name, module in [('scan_engine', engine), ('vector_store', vector), ('bec_detector', bec)]:
        monkeypatch.setitem(sys.modules, 'services.' + name, module)
        monkeypatch.setattr(services, name, module, raising=False)
    return calls


def test_preload_is_sequential_and_reports_observed_availability(monkeypatch):
    calls = fake_components(monkeypatch)
    result = preload_email_components()
    assert calls == ['models', 'embedding', 'collection', 'bec']
    assert result['components_available'] is True
    assert all(s['available'] and s['wall_seconds'] >= 0 for s in result['stages'])


def test_missing_preload_component_does_not_claim_readiness(monkeypatch):
    fake_components(monkeypatch, unavailable_bec=True)
    result = preload_email_components()
    assert result['components_available'] is False
    assert result['stages'][-1]['available'] is False
    assert 'unavailable' in result['stages'][-1]['error']
