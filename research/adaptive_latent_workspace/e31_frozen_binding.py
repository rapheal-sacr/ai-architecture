"""Frozen E31 tutorial and renaming diagnostic; no operational learning."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,subprocess,time
import numpy as np,torch
from delivery_world import identity,canonical
from delivery_diagnostics import generate,rename
from language_delivery_agent import LanguageActor
from preflight_language_adapter import base_hash,adapter_state
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();cfg=json.loads((HERE/'protocol_e31.json').read_text());source_path=a.source/'complete.json';source_hash=sha(source_path);source=json.loads(source_path.read_text());scfg=source['manifest']['config'];assert source['status']==('PREFLIGHT_ONLY' if a.preflight else cfg['source_status'])
    if not a.preflight:assert source['manifest']['git_commit'].startswith(cfg['source_commit']) and scfg['seeds']==cfg['source_seeds']
    else:cfg['source_seeds']=scfg['seeds']
    torch.set_num_threads(scfg['torch_threads']);torch.use_deterministic_algorithms(True);assert sha(a.model/'model.safetensors')==scfg['base_weights_sha256'];a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'complete.json').exists()
    files=['protocol_e31.json','e31_frozen_binding.py','delivery_diagnostics.py','delivery_world.py','language_delivery_agent.py','preflight_language_adapter.py','e28_depth.py'];manifest=dict(config=cfg,preflight=a.preflight,source_config=scfg,source_complete_sha256=source_hash,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_identity={f:sha(HERE/f) for f in files},source_manifest=source['manifest']);(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');datasets=[];results=[]
    for seed in cfg['source_seeds']:
        boot=next(b for b in source['bootstrap'] if b['seed']==seed);final=next(c for c in source['results'] if c['seed']==seed and c['arm']=='adaptive_bounded')['episodes'][-1];assert sha(boot['checkpoint'])==boot['checkpoint_sha256'] and sha(final['checkpoint'])==final['checkpoint_sha256'];start=time.perf_counter();queries,generation=generate(seed,boot,scfg,cfg,a.preflight);views={encoding:[rename(q,encoding,seed+cfg['renaming_seed_offset']+i*1009) for i,q in enumerate(queries)] for encoding in cfg['encodings']};generation['seconds']=time.perf_counter()-start;dataset=dict(seed=seed,queries=queries,views=views,generation=generation,query_source_hash=identity(queries),source_bootstrap=boot['checkpoint_sha256'],source_final=final['checkpoint_sha256'],inherited_bootstrap={k:boot[k] for k in ('teacher_actions','training_seconds','model_load_seconds','work')});dp=a.out/f'{seed}-queries.json';dp.write_text(json.dumps(dataset,indent=2)+'\n');datasets.append(dict(seed=seed,path=str(dp),sha256=sha(dp),generation=generation));actor=LanguageActor(a.model,scfg,seed);base=base_hash(actor.model);assert base==boot['base_hash'];boot_state=torch.load(boot['checkpoint'],map_location='cpu',weights_only=False);final_state=torch.load(final['checkpoint'],map_location='cpu',weights_only=False)['actor']
        for state in cfg['states']:
            if state=='base':assert all(torch.count_nonzero(v)==0 for k,v in adapter_state(actor.model).items() if k.endswith('.b'))
            else:actor.load_adapter((boot_state if state=='bootstrap' else final_state)['adapter'])
            actor_before=actor.state();actor_before.pop('work');before_hash=state_hash(actor_before);adapter_hash=state_hash(adapter_state(actor.model));state_start=time.perf_counter()
            for encoding in cfg['encodings']:
                work_before=copy.deepcopy(actor.work);torch.cuda.synchronize();start=time.perf_counter();rows=[];torch.cuda.reset_peak_memory_stats()
                for qi,(q,view) in enumerate(zip(queries,views[encoding])):
                    with torch.no_grad():logits,tokens=actor.logits(view['prompt'],view['actions'],'inference');logp=torch.log_softmax(logits,-1);probs=.95*torch.softmax(logits,-1)+.05/len(view['actions']);prediction=int(torch.argmax(logits));allowed=q['reference']['indices'];row=dict(index=qi,split=q['split'],category=q['reference']['category'],reference_indices=allowed,teacher_index=q['teacher_index'],prediction=prediction,rule_correct=prediction in allowed,teacher_exact=prediction==q['teacher_index'],teacher_cross_entropy=float(-logp[q['teacher_index']]),rule_probability=float(probs[allowed].sum()),tokens=tokens,logits=logits.cpu().tolist(),mapping_utf8_bytes=len(canonical(view['mapping']).encode()),bootstrap_index_exposures=q['bootstrap_index_exposures'],bootstrap_prompt_label_exposures=q['bootstrap_prompt_label_exposures']);rows.append(row)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start;work={k:actor.work[k]-work_before[k] for k in actor.work};after=actor.state();after.pop('work');assert state_hash(after)==before_hash;result=dict(seed=seed,state=state,encoding=encoding,rows=rows,query_seconds=elapsed,model_load_seconds=actor.load_seconds,work=work,costs=actor.costs(),cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),adapter_sha256=adapter_hash,actor_weights_optimizer_memory_rng_unchanged=True,dataset_sha256=sha(dp));results.append(result);(a.out/f'{seed}-{state}-{encoding}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='E31_CELL',seed=seed,state=state,encoding=encoding,queries=len(rows),correct=sum(r['rule_correct'] for r in rows),seconds=elapsed)),flush=True)
            assert base_hash(actor.model)==base
        assert sha(boot['checkpoint'])==boot['checkpoint_sha256'] and sha(final['checkpoint'])==final['checkpoint_sha256'];del actor;gc.collect();torch.cuda.empty_cache()
    assert sha(source_path)==source_hash;complete=dict(status='PREFLIGHT_ONLY' if a.preflight else 'E31_COMPLETE',manifest=manifest,datasets=datasets,results=results);(a.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],cells=len(results))),flush=True)
if __name__=='__main__':main()
