import json

import pytest

from utils.jsonl import read_jsonl


def test_unicode_line_characters_are_body_content_not_new_records(tmp_path):
    rows = [{'body': 'first\u0085second\u2028third\u2029fourth'}, {'body':'next email'}]
    path = tmp_path / 'messages.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    assert read_jsonl(path) == rows


def test_crlf_records_and_blank_lines_do_not_change_body(tmp_path):
    path = tmp_path / 'messages.jsonl'
    path.write_bytes(b'{"body":"line1\\nline2"}\r\n\r\n{"body":"two"}\r\n')
    assert read_jsonl(path) == [{'body':'line1\nline2'}, {'body':'two'}]


@pytest.mark.parametrize('invalid', ['{"p": NaN}', '{"p": Infinity}', '{broken'])
def test_invalid_record_is_not_silently_dropped(tmp_path, invalid):
    path = tmp_path / 'messages.jsonl'
    path.write_text('{"body":"ok"}\n' + invalid + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='JSONL record 2'):
        read_jsonl(path)
