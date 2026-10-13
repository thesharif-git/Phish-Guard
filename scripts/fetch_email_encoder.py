"""Download only pinned public E5 files; no remote code is executed."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '614241f622f53c4eeff9890bdc4f31cfecc418b3'
REPOSITORY = 'intfloat/multilingual-e5-small'
FILES = ('onnx/model_qint8_avx512_vnni.onnx', 'tokenizer.json', 'config.json',
         'tokenizer_config.json')


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = args.output.resolve()
    destination.relative_to(root / 'benchmarks')
    destination.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true'
    with urllib.request.urlopen(url, timeout=60) as response:
        metadata = json.load(response)
    if metadata['sha'] != REVISION:
        raise ValueError('Repository revision mismatch')
    inventory = {item['rfilename']: item for item in metadata['siblings']}
    manifest = dict(repository=REPOSITORY, revision=REVISION, files={}, complete=False)
    for name in FILES:
        item = inventory[name]
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item.get('lfs', {}).get('sha256')
        if not (target.exists() and target.stat().st_size == item['size'] and
                (not expected or sha256(target) == expected)):
            partial = target.with_suffix(target.suffix + '.download')
            with urllib.request.urlopen(f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}',
                                        timeout=120) as response, partial.open('wb') as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
            if partial.stat().st_size != item['size'] or (expected and sha256(partial) != expected):
                raise ValueError('Downloaded file differs from publisher: ' + name)
            if not expected:
                git_blob = hashlib.sha1(b'blob ' + str(item['size']).encode() + b'\0' + partial.read_bytes()).hexdigest()
                if git_blob != item['blobId']:
                    raise ValueError('Git blob mismatch: ' + name)
            partial.replace(target)
        manifest['files'][name] = dict(bytes=target.stat().st_size, sha256=sha256(target),
                                      publisher_sha256=expected, publisher_git_blob=item['blobId'])
        print(json.dumps(dict(file=name, bytes=target.stat().st_size, verified=True)), flush=True)
    manifest['complete'] = True
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
