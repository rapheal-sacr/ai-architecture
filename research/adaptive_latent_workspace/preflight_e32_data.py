"""Disjoint full-sized development corpora, with independent reference checks."""
from pathlib import Path
import json,hashlib,collections
from delivery_training_data import make_data,assert_disjoint
from record_e31 import independent_reference

def main():
    cfg=json.loads(Path('protocol_e32.json').read_text());data=[make_data(seed,cfg) for seed in (1911,1912,1913)];checks=assert_disjoint(data)
    for d in data:
        for q in d['queries']:assert independent_reference(q)==q['reference']['indices']
        assert len(d['order'])==1024
    result=dict(status='E32_FULL_DATA_PREFLIGHT_ONLY',seeds=[d['seed'] for d in data],disjoint=checks,cases=[dict(seed=d['seed'],teacher_actions=d['teacher_actions'],queries=len(d['queries']),categories=dict(collections.Counter(q['reference']['category'] for q in d['queries'])),example_sha256=d['example_sha256'],order_sha256=d['order_sha256']) for d in data],references_independently_verified=True,source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('delivery_training_data.py','preflight_e32_data.py')});Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e32-data-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
