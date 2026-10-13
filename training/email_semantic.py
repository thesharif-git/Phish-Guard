"""Frozen multilingual Email encoder for research, not a production approval.

Canonical content matches the existing text candidate. No labels, source IDs,
annotation fields, sender claims or publisher explanations enter embeddings.
Long messages use all non-overlapping token windows; none are silently dropped.
"""
from pathlib import Path
import time

import numpy as np

from utils.email_candidate_text import canonical_email_candidate_text, EMAIL_CANDIDATE_TEXT_VERSION
from utils.email_input_contract import validate_email_input

SEMANTIC_TEXT_VERSION = 'email-canonical-content-e5-query-v1:' + EMAIL_CANDIDATE_TEXT_VERSION


def content_text(record):
    record = validate_email_input(record)
    body = canonical_email_candidate_text(record['body'])
    if not body:
        body = canonical_email_candidate_text(record['html_body'])
    return (body + ' ' + canonical_email_candidate_text(record['subject'])).strip()


def masked_mean(hidden, mask):
    hidden, mask = np.asarray(hidden), np.asarray(mask)
    if hidden.ndim != 3 or mask.shape != hidden.shape[:2] or not np.isfinite(hidden).all():
        raise ValueError('Invalid encoder hidden states or attention mask')
    counts = mask.sum(axis=1)
    if (counts <= 0).any():
        raise ValueError('Empty encoder attention mask')
    result = (hidden * mask[..., None]).sum(axis=1) / counts[:, None]
    norm = np.linalg.norm(result, axis=1, keepdims=True)
    if not np.isfinite(result).all() or (norm <= 0).any():
        raise ValueError('Invalid or zero encoder embedding')
    return (result / norm).astype(np.float32)


def token_windows(ids, *, bos, eos, max_tokens=512):
    if max_tokens < 4 or len(ids) < 2 or ids[0] != bos or ids[-1] != eos:
        raise ValueError('Unexpected tokenizer special-token contract')
    payload = list(ids[1:-1])
    if not payload:
        raise ValueError('No message tokens')
    width = max_tokens - 2
    return [[bos] + payload[start:start + width] + [eos]
            for start in range(0, len(payload), width)]


class FrozenEmailEncoder:
    def __init__(self, directory, *, max_tokens=512, threads=2):
        # Load only local tokenizer JSON and ONNX, never remote Python code.
        import onnxruntime as ort
        from tokenizers import Tokenizer
        directory = Path(directory)
        self.tokenizer = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.bos = self.tokenizer.token_to_id('<s>')
        self.eos = self.tokenizer.token_to_id('</s>')
        self.pad = self.tokenizer.token_to_id('<pad>')
        if None in (self.bos, self.eos, self.pad):
            raise ValueError('Missing XLM-R tokenizer special tokens')
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(directory / 'onnx/model_qint8_avx512_vnni.onnx'),
                                            sess_options=options, providers=['CPUExecutionProvider'])
        self.inputs = {item.name for item in self.session.get_inputs()}
        if not {'input_ids', 'attention_mask'} <= self.inputs or self.inputs - {'input_ids', 'attention_mask', 'token_type_ids'}:
            raise ValueError('Unexpected ONNX input contract')
        self.max_tokens = max_tokens
        self.threads = threads

    def encode(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('No observed email content')
        started = time.perf_counter()
        ids = self.tokenizer.encode('query: ' + text).ids
        windows = token_windows(ids, bos=self.bos, eos=self.eos, max_tokens=self.max_tokens)
        vectors, weights = [], []
        for window in windows:
            values = np.asarray([window], dtype=np.int64)
            mask = np.ones_like(values)
            feed = {'input_ids': values, 'attention_mask': mask}
            if 'token_type_ids' in self.inputs:
                feed['token_type_ids'] = np.zeros_like(values)
            hidden = self.session.run(None, feed)[0]
            vectors.append(masked_mean(hidden, mask)[0])
            weights.append(len(window) - 2)
        vector = np.average(np.asarray(vectors), axis=0, weights=weights)
        norm = np.linalg.norm(vector)
        if not np.isfinite(vector).all() or norm <= 0:
            raise ValueError('Invalid aggregated message embedding')
        return (vector / norm).astype(np.float32), dict(
            total_tokens=len(ids), windows=len(windows), dropped_content_tokens=0,
            wall_ms=(time.perf_counter() - started) * 1000)
