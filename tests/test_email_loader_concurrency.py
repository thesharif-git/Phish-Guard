"""Exercise first-load races with fake dependencies, never real model weights."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import sys
import threading
import time
import types
import unittest
from unittest.mock import patch

from services import bec_detector, vector_store


def simultaneous_calls(function, workers=8):
    start = threading.Barrier(workers + 1)

    def call():
        start.wait(timeout=5)
        return function()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(call) for _ in range(workers)]
        start.wait(timeout=5)
        return [future.result(timeout=5) for future in futures]


class EmailLoaderConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.saved_bec = (bec_detector._BEC_PIPELINE, bec_detector._PIPELINE_LOADED)
        self.saved_vector = (vector_store._EMBEDDING_MODEL, vector_store._CHROMADB_CLIENT,
                             vector_store._COLLECTION, vector_store._INITIALIZED)
        bec_detector._BEC_PIPELINE = None
        bec_detector._PIPELINE_LOADED = False
        vector_store._EMBEDDING_MODEL = None
        vector_store._CHROMADB_CLIENT = None
        vector_store._COLLECTION = None
        vector_store._INITIALIZED = False

    def tearDown(self):
        bec_detector._BEC_PIPELINE, bec_detector._PIPELINE_LOADED = self.saved_bec
        (vector_store._EMBEDDING_MODEL, vector_store._CHROMADB_CLIENT,
         vector_store._COLLECTION, vector_store._INITIALIZED) = self.saved_vector

    def fake_transformers(self, unavailable=()):
        calls = {'tokenizer': [], 'model': [], 'pipeline': []}
        module = types.ModuleType('transformers')

        def tokenizer(name, **kwargs):
            self.assertEqual(kwargs, {'local_files_only': True})
            calls['tokenizer'].append(name)
            time.sleep(0.03)  # Hold the first loader open while all callers arrive.
            if name in unavailable:
                raise OSError('Fake locally unavailable weights')
            return object()

        def model(name, **kwargs):
            self.assertEqual(kwargs, {'local_files_only': True})
            calls['model'].append(name)
            return object()

        def pipeline(task, **kwargs):
            self.assertEqual(task, 'zero-shot-classification')
            self.assertEqual(kwargs['device'], -1)
            instance = object()
            calls['pipeline'].append(instance)
            return instance

        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=tokenizer)
        module.AutoModelForSequenceClassification = types.SimpleNamespace(from_pretrained=model)
        module.pipeline = pipeline
        return module, calls

    def test_bec_primary_constructed_once(self):
        module, calls = self.fake_transformers()
        with patch.dict(sys.modules, {'transformers': module}), \
                patch.object(bec_detector.settings, 'BEC_MODEL', 'test/primary'):
            results = simultaneous_calls(bec_detector._get_bec_classifier)
            self.assertIs(bec_detector._get_bec_classifier(), results[0])
        self.assertEqual(calls['tokenizer'], ['test/primary'])
        self.assertEqual(calls['model'], ['test/primary'])
        self.assertEqual(len(calls['pipeline']), 1)
        self.assertTrue(all(result is results[0] for result in results))

    def test_bec_fallback_constructed_once(self):
        module, calls = self.fake_transformers(unavailable=('test/primary',))
        with patch.dict(sys.modules, {'transformers': module}), \
                patch.object(bec_detector.settings, 'BEC_MODEL', 'test/primary'), \
                patch.object(bec_detector.settings, 'FALLBACK_BEC_MODEL', 'test/fallback'):
            results = simultaneous_calls(bec_detector._get_bec_classifier)
        self.assertEqual(calls['tokenizer'], ['test/primary', 'test/fallback'])
        self.assertEqual(calls['model'], ['test/fallback'])
        self.assertEqual(len(calls['pipeline']), 1)
        self.assertTrue(all(result is results[0] for result in results))

    def test_bec_unavailable_result_is_cached(self):
        module, calls = self.fake_transformers(unavailable=('test/primary', 'test/fallback'))
        with patch.dict(sys.modules, {'transformers': module}), \
                patch.object(bec_detector.settings, 'BEC_MODEL', 'test/primary'), \
                patch.object(bec_detector.settings, 'FALLBACK_BEC_MODEL', 'test/fallback'):
            self.assertEqual(simultaneous_calls(bec_detector._get_bec_classifier), [None] * 8)
            self.assertIsNone(bec_detector._get_bec_classifier())
        self.assertEqual(calls['tokenizer'], ['test/primary', 'test/fallback'])
        self.assertTrue(bec_detector._PIPELINE_LOADED)

    def fake_embeddings(self, unavailable=()):
        calls = []
        module = types.ModuleType('sentence_transformers')

        class FakeEmbedding:
            def encode(self, documents):
                return types.SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in documents])

        def create(name, **kwargs):
            self.assertEqual(kwargs, {'local_files_only': True})
            calls.append(name)
            time.sleep(0.03)
            if name in unavailable:
                raise OSError('Fake locally unavailable weights')
            return FakeEmbedding()

        module.SentenceTransformer = create
        return module, calls

    def check_embedding_load(self, unavailable, expected_names, available):
        module, calls = self.fake_embeddings(unavailable)
        with patch.dict(sys.modules, {'sentence_transformers': module}), \
                patch.object(vector_store.settings, 'VECTOR_EMBEDDING_MODEL', 'test/primary'), \
                patch.object(vector_store.settings, 'FALLBACK_VECTOR_MODEL', 'test/fallback'):
            results = simultaneous_calls(vector_store._get_embedding_model)
            self.assertIs(vector_store._get_embedding_model(), results[0])
        self.assertEqual(calls, expected_names)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertEqual(results[0] is not None, available)

    def test_embedding_primary_constructed_once(self):
        self.check_embedding_load((), ['test/primary'], True)

    def test_embedding_fallback_constructed_once(self):
        self.check_embedding_load(('test/primary',), ['test/primary', 'test/fallback'], True)

    def test_embedding_unavailable_result_is_cached(self):
        self.check_embedding_load(('test/primary', 'test/fallback'),
                                  ['test/primary', 'test/fallback'], False)
        self.assertIs(vector_store._EMBEDDING_MODEL, False)

    def fake_chroma(self, first_count_gate=None):
        calls = {'client': 0, 'collection': 0, 'seed': 0}
        module = types.ModuleType('chromadb')

        class Collection:
            first_count = True

            def count(self):
                if self.first_count and first_count_gate is not None:
                    self.first_count = False
                    entered, release = first_count_gate
                    entered.set()
                    if not release.wait(timeout=5):
                        raise TimeoutError('Test did not release initialization')
                return 0

            def add(self, **kwargs):
                calls['seed'] += 1

        collection = Collection()

        class Client:
            def get_or_create_collection(self, **kwargs):
                calls['collection'] += 1
                return collection

        def create(**kwargs):
            calls['client'] += 1
            time.sleep(0.03)
            return Client()

        module.PersistentClient = create
        return module, calls, collection

    def test_chroma_constructs_and_seeds_once_without_nested_loader_deadlock(self):
        chroma, calls, collection = self.fake_chroma()
        embeddings, embedding_calls = self.fake_embeddings()
        with patch.dict(sys.modules, {'chromadb': chroma, 'sentence_transformers': embeddings}), \
                patch.object(vector_store.os, 'makedirs'), \
                patch.object(vector_store.settings, 'VECTOR_EMBEDDING_MODEL', 'test/primary'):
            results = simultaneous_calls(vector_store._get_chroma_collection)
        self.assertTrue(all(result is collection for result in results))
        self.assertEqual(calls, {'client': 1, 'collection': 1, 'seed': 1})
        self.assertEqual(embedding_calls, ['test/primary'])
        self.assertTrue(vector_store._INITIALIZED)

    def test_chroma_waits_until_initialization_finishes(self):
        entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
        chroma, calls, collection = self.fake_chroma((entered, release))
        embeddings, _ = self.fake_embeddings()

        def second_load():
            second_started.set()
            return vector_store._get_chroma_collection()

        with patch.dict(sys.modules, {'chromadb': chroma, 'sentence_transformers': embeddings}), \
                patch.object(vector_store.os, 'makedirs'):
            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(vector_store._get_chroma_collection)
                try:
                    self.assertTrue(entered.wait(timeout=5))
                    second = pool.submit(second_load)
                    self.assertTrue(second_started.wait(timeout=5))
                    with self.assertRaises(TimeoutError):
                        second.result(timeout=0.05)
                finally:
                    release.set()
                self.assertIs(first.result(timeout=5), collection)
                self.assertIs(second.result(timeout=5), collection)
        self.assertEqual(calls, {'client': 1, 'collection': 1, 'seed': 1})

    def test_chroma_initialization_failure_remains_retryable(self):
        chroma = types.ModuleType('chromadb')
        chroma.PersistentClient = unittest.mock.Mock(side_effect=OSError('Fake store unavailable'))
        with patch.dict(sys.modules, {'chromadb': chroma}), patch.object(vector_store.os, 'makedirs'):
            self.assertIsNone(vector_store._get_chroma_collection())
            self.assertIsNone(vector_store._get_chroma_collection())
        self.assertEqual(chroma.PersistentClient.call_count, 2)
        self.assertFalse(vector_store._INITIALIZED)

    def test_chroma_count_failure_clears_partial_state_and_retries(self):
        chroma = types.ModuleType('chromadb')
        failed_collection = types.SimpleNamespace(
            count=unittest.mock.Mock(side_effect=OSError('Fake collection count failed')))
        ready_collection = types.SimpleNamespace(count=unittest.mock.Mock(return_value=0),
                                                 add=unittest.mock.Mock())
        failed_client = types.SimpleNamespace(get_or_create_collection=unittest.mock.Mock(
            return_value=failed_collection))
        ready_client = types.SimpleNamespace(get_or_create_collection=unittest.mock.Mock(
            return_value=ready_collection))
        chroma.PersistentClient = unittest.mock.Mock(side_effect=[failed_client, ready_client])
        embeddings, embedding_calls = self.fake_embeddings()
        with patch.dict(sys.modules, {'chromadb': chroma, 'sentence_transformers': embeddings}), \
                patch.object(vector_store.os, 'makedirs'):
            self.assertIsNone(vector_store._get_chroma_collection())
            self.assertIsNone(vector_store._CHROMADB_CLIENT)
            self.assertIsNone(vector_store._COLLECTION)
            self.assertFalse(vector_store._INITIALIZED)
            self.assertIs(vector_store._get_chroma_collection(), ready_collection)
            self.assertIs(vector_store._get_chroma_collection(), ready_collection)
        self.assertEqual(chroma.PersistentClient.call_count, 2)
        self.assertIs(vector_store._CHROMADB_CLIENT, ready_client)
        self.assertTrue(vector_store._INITIALIZED)
        self.assertEqual(failed_collection.count.call_count, 1)
        ready_collection.add.assert_called_once()
        self.assertEqual(len(embedding_calls), 1)


if __name__ == '__main__':
    unittest.main()
