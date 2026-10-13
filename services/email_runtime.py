"""Optional sequential initialization for a local Email scanning worker.

Preloading moves initialization before the first request. It does not reduce
the total initialization cost or establish model accuracy/calibration.
An empty campaign collection retains the existing initialization behavior:
it computes and stores embeddings of the configured reference examples.
"""
import time


def preload_email_components():
    from services import scan_engine, vector_store, bec_detector

    started = time.perf_counter()
    stages = []

    def load(name, function, available):
        wall, cpu = time.perf_counter(), time.process_time()
        stage = {'component': name, 'available': False}
        try:
            value = function()
            stage['available'] = bool(available(value))
            if not stage['available']:
                stage['error'] = 'Component unavailable after configured local-cache fallback'
        except Exception as e:
            stage['error'] = type(e).__name__ + ': ' + str(e)
        finally:
            stage['wall_seconds'] = time.perf_counter() - wall
            stage['process_cpu_seconds'] = time.process_time() - cpu
            stages.append(stage)

    load('content_and_embedded_link_models', scan_engine.load_ml_model,
         lambda _: all(m is not None for m in (scan_engine.ML_MODEL, scan_engine.URL_MODEL, scan_engine.DOMAIN_MODEL)))
    load('campaign_embedding', vector_store._get_embedding_model, lambda model: model is not None)
    load('campaign_collection', vector_store._get_chroma_collection, lambda collection: collection is not None)
    load('bec_classifier', bec_detector._get_bec_classifier, lambda classifier: classifier is not None)
    return {'preloaded': True, 'components_available': all(s['available'] for s in stages),
            'wall_seconds': time.perf_counter() - started, 'stages': stages,
            'scope': 'Component availability only; not generalization or calibration approval'}
