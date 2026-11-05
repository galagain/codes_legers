#!/usr/bin/env python3
"""Check syntax, weight integrity and internal symlinks (standard library only)."""
import ast
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
count = 0
for path in root.rglob('*.py'):
    ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    count += 1
for path in root.rglob('*'):
    if path.is_symlink():
        assert path.exists(), f'Broken link: {path}'
        assert path.resolve().is_relative_to(root), f'External link: {path}'
manifest = json.loads((root / 'weights/manifest.json').read_text())
for entry in manifest:
    path = root / entry['file']
    assert path.stat().st_size == entry['bytes'], path
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    assert digest.hexdigest() == entry['sha256'], path
    if 'config' in entry:
        assert (root / entry['config']).is_file(), entry['config']
examples = json.loads((root / 'examples/provenance.json').read_text())
for entry in examples:
    path = root / 'examples' / entry['file']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], path
print(f'OK: {count} Python files, {len(manifest)} weight files, {len(examples)} examples, internal links.')
