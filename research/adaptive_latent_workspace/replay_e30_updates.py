"""Reexecute E30 online updates from actual recorded experience, audit-only."""
from pathlib import Path
import argparse,gc,hashlib,json,time
import torch
from language_delivery_agent import LanguageActor
from preflight_language_adapter import base_hash,adapter_state
from e28_depth import state_hash

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();source_path=a.source/'complete.json';source_hash=sha(source_path);data=json.loads(source_path.read_text());assert data['status'] in ('E30_COMPLETE','PREFLIGHT_ONLY');cfg=data['manifest']['config'];torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True);results=[]
    for case in data['results']:
        if case['arm']!='adaptive_bounded':continue
        bootstrap=next(b for b in data['bootstrap'] if b['seed']==case['seed']);assert sha(bootstrap['checkpoint'])==bootstrap['checkpoint_sha256'];boot=torch.load(bootstrap['checkpoint'],map_location='cpu',weights_only=False);actor=LanguageActor(a.model,cfg,case['seed']);actor.load_adapter(boot['adapter']);actor.start_optimizer(cfg['online_learning_rate'],boot['optimizer']);before=base_hash(actor.model);assert before==bootstrap['base_hash'];torch.cuda.synchronize();start=time.perf_counter();audits=[]
        for ep in case['episodes']:
            updates=actor.observe_return(ep['trajectory'],ep['success']);assert updates==ep['updates'];assert sha(ep['checkpoint'])==ep['checkpoint_sha256'];saved=torch.load(ep['checkpoint'],map_location='cpu',weights_only=False)['actor'];actual=actor.state()
            for k in ('adapter','optimizer','replay','seen','learning_rng','torch_rng','cuda_rng'):assert state_hash(actual[k])==state_hash(saved[k]),(case['seed'],ep['mission'],k)
            for k in ('update_calls','update_tokens','update_attention_elements'):assert actual['work'][k]==saved['work'][k]
            audits.append(dict(mission=ep['mission'],adapter_optimizer_replay_rng_losses_exact=True,checkpoint_sha256=ep['checkpoint_sha256']));print(json.dumps(dict(event='E30_UPDATE_REPLAY_AUDIT',seed=case['seed'],mission=ep['mission'],exact=True)),flush=True)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;assert base_hash(actor.model)==before;results.append(dict(seed=case['seed'],missions=audits,audit_seconds=seconds,audit_work=actor.costs(),base_unchanged=True));del actor;gc.collect();torch.cuda.empty_cache()
    assert sha(source_path)==source_hash;result=dict(status='E30_UPDATE_REPLAY_AUDITED',source_status=data['status'],source_sha256=source_hash,script_sha256=sha(__file__),results=results,scope='All adaptive online updates reexecuted from actual stored trajectories and bootstrap optimizer. Every mission adapter, optimizer, replay, learning RNG, torch/CUDA RNG, losses and update work exactly match. No counterfactual policy actions or new environment outcomes; this duplicate audit work is excluded from operational policy costs and reported separately. Bootstrap fitting and operational logits are not reexecuted.');a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],cases=len(results))))
if __name__=='__main__':main()
