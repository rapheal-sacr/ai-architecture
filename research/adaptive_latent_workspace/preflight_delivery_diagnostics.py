"""Structural E31 checks: tied routes and irrelevant-state noninterference."""
from pathlib import Path
import copy,json,hashlib
from delivery_world import ObservationMemory,render_prompt
from delivery_diagnostics import reference,query,rename

def main():
    edges={'raaaaa':['rbbbbb','rccccc'],'rbbbbb':['raaaaa','rddddd'],'rccccc':['raaaaa','rddddd'],'rddddd':['rbbbbb','rccccc']};mem=ObservationMemory();obs=None
    for room in ['rddddd','rccccc','rbbbbb','raaaaa']:
        obs=dict(world='waaaaa',room=room,home='rddddd',target='oaaaaa',carrying='oaaaaa',exits=edges[room],items=[],feedback='Moved to '+room+'.',done=False);mem.observe(obs)
    q=query(obs,mem,'move:rbbbbb','structural',0);assert q['reference']==dict(category='home_multi_edge',indices=[0,1],distance=2)
    checks=[]
    for encoding in ('original','fresh_nonce','compact'):
        transformed=rename(q,encoding,5432);checks.append(dict(encoding=encoding,reference=transformed['reference'],prompt=transformed['prompt'],mapping=transformed['mapping']))
    extra=copy.deepcopy(q);extra['memory']['records'].append(dict(world='wzzzzz',room='rzzzzz',exits=['ryyyyy'],items=['ozzzzz']));extra['memory']['writes']+=1
    for encoding in ('fresh_nonce','compact'):assert rename(extra,encoding,5432)==rename(q,encoding,5432)
    # A one-edge home route has exactly one reference action even when another exit exists.
    short=copy.deepcopy(obs);short['home']='rbbbbb';shortq=query(short,mem,'move:rbbbbb','structural',1);assert shortq['reference']==dict(category='home_one_edge',indices=[0],distance=1)
    for encoding in ('fresh_nonce','compact'):assert rename(shortq,encoding,4321)['reference']==shortq['reference']
    result=dict(status='E31_STRUCTURAL_PREFLIGHT_ONLY',tied_routes_accepted=True,one_edge_route_unique=True,irrelevant_world_state_does_not_change_mapping=True,checks=checks,source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('delivery_diagnostics.py','preflight_delivery_diagnostics.py')});Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e31-structural-preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
if __name__=='__main__':main()
