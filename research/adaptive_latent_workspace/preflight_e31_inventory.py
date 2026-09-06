"""Full development inventory without neural queries or scored source seeds."""
from pathlib import Path
import json,hashlib,collections
from transformers import AutoTokenizer
from delivery_diagnostics import generate,rename
from delivery_world import RULES
ROOT=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research')
def main():
    cfg=json.loads(Path('protocol_e31.json').read_text());source=json.loads((ROOT/'alw-runs/e30-preflight-registered/complete.json').read_text());boot=source['bootstrap'][0];assert boot['seed']==1711;queries,generation=generate(1711,boot,source['manifest']['config'],cfg,False);tok=AutoTokenizer.from_pretrained(ROOT/'models/qwen2.5-0.5b-instruct-7ae5576',local_files_only=True);counts={}
    for encoding in cfg['encodings']:
        lengths=[]
        for i,q in enumerate(queries):
            view=rename(q,encoding,1711+cfg['renaming_seed_offset']+i*1009);text=tok.apply_chat_template([dict(role='system',content=RULES),dict(role='user',content=view['prompt'])],tokenize=False,add_generation_prompt=True);lengths.append(len(tok(text)['input_ids']))
        assert max(lengths)<=source['manifest']['config']['max_prompt_tokens'];counts[encoding]=dict(queries=len(lengths),min_tokens=min(lengths),max_tokens=max(lengths),total_tokens=sum(lengths))
    result=dict(status='E31_FULL_DEVELOPMENT_INVENTORY_ONLY',seed=1711,generation=generation,categories=dict(collections.Counter(q['reference']['category'] for q in queries)),encodings=counts,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(ROOT/'alw-runs/e31-inventory-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
