"""Check integrity and provenance, not correctness of upstream ML implementations."""
from pathlib import Path
import json,hashlib,ast
ROOT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'docs/source-manifest.json').read_text(encoding='utf8'))
for item in manifest['files']:
    path=ROOT/item['local_path']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==item['local_sha256'],str(path)
count=0
for path in (ROOT/'notebooks').rglob('*.ipynb'):
    nb=json.loads(path.read_text(encoding='utf8'));assert nb['nbformat']==4
    ids=set()
    for c in nb['cells']:
        assert c['id'] not in ids;ids.add(c['id'])
        assert c['cell_type'] in ['markdown','code','raw']
        if c['cell_type']=='code':assert c['execution_count'] is None and c['outputs']==[]
    count+=1
for path in ROOT.rglob('*.py'):ast.parse(path.read_text(encoding='utf8'),filename=str(path))
print(f'PASS: {count} notebooks, {len(manifest["files"])} source hashes; Python syntax valid.')
