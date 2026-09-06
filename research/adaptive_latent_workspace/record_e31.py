"""Reconstruct E31 queries, verify reference sets via Floyd-Warshall, audit metrics."""
from pathlib import Path
import argparse,collections,hashlib,json,math,subprocess,time
import numpy as np,torch
from transformers import AutoTokenizer
from delivery_diagnostics import generate,rename
from delivery_world import canonical,identity,RULES
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def independent_reference(q):
    obs=q['observation'];actions=q['actions'];here=obs['room'];target=obs['target'];known={r['room']:r for r in q['memory']['records'] if r['world']==obs['world']}
    assert q['prompt'].count('Options:\n')==1;option_lines=q['prompt'].split('Options:\n')[1].splitlines()[:-1];assert [line.split('. ',1)[1] for line in option_lines]==actions
    memory_lines=q['prompt'].split('Current world: ')[0].splitlines()[1:];expected=[r+' | exits '+','.join(d['exits'])+' | stock '+(','.join(d['items']) or 'none') for r,d in known.items()];assert memory_lines==expected
    if obs['carrying']==target and here==obs['home']:return [actions.index('deliver:'+target)]
    if obs['carrying'] not in (None,target):return [actions.index('drop:'+obs['carrying'])]
    if obs['carrying'] is None and target in obs['items']:return [actions.index('pick:'+target)]
    nodes=set(known)
    for d in known.values():nodes.update(d['exits'])
    nodes=sorted(nodes);n=len(nodes);at={r:i for i,r in enumerate(nodes)};dist=np.full((n,n),1e6);np.fill_diagonal(dist,0)
    for room,d in known.items():
        for other in d['exits']:dist[at[room],at[other]]=dist[at[other],at[room]]=1
    for k in range(n):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    if obs['carrying']==target:goals=[obs['home']]
    else:
        goals=[r for r,d in known.items() if target in d['items']]
        if not goals:goals=[r for r in nodes if r not in known]
        if not goals:goals=[r for r in known if r!=here][:1]
    goals=[g for g in goals if g in at]
    if goals and here in at:
        remaining={r:min(dist[at[r],at[g]] for g in goals) for r in nodes};d=remaining[here]
        if 0<d<1e6:return [i for i,a in enumerate(actions) if a.startswith('move:') and remaining.get(a.split(':')[1],1e6)==d-1]
    return [next((i for i,a in enumerate(actions) if a.startswith('move:')),actions.index('wait:here'))]

def summarize(rows):
    return dict(n=len(rows),rule_correct=sum(r['rule_correct'] for r in rows),rule_accuracy=float(np.mean([r['rule_correct'] for r in rows])),teacher_exact=sum(r['teacher_exact'] for r in rows),teacher_accuracy=float(np.mean([r['teacher_exact'] for r in rows])),mean_teacher_ce=float(np.mean([r['teacher_cross_entropy'] for r in rows])),mean_rule_probability=float(np.mean([r['rule_probability'] for r in rows])),tokens=sum(r['tokens'] for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--parent',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1);path=a.source/'complete.json';before=sha(path);data=json.loads(path.read_text());assert data['status']==('PREFLIGHT_ONLY' if a.preflight else 'E31_COMPLETE');manifest=data['manifest'];cfg=manifest['config'];scfg=manifest['source_config'];parent=json.loads((a.parent/'complete.json').read_text());assert sha(a.parent/'complete.json')==manifest['source_complete_sha256'];assert parent['manifest']['config']==scfg
    if not a.preflight:assert manifest['git_commit'].startswith('ff78ad7')
    for f,h in manifest['code_identity'].items():
        assert sha(HERE/f)==h
        if not a.preflight:assert hashlib.sha256(subprocess.check_output(['git','show',manifest['git_commit']+':research/adaptive_latent_workspace/'+f],cwd=HERE)).hexdigest()==h
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);cache={};datasets={};source_adapters={};inventory=[];allsummary=[];pairs=[]
    for ds in data['datasets']:
        assert sha(ds['path'])==ds['sha256'];d=json.loads(Path(ds['path']).read_text());seed=d['seed'];boot=next(b for b in parent['bootstrap'] if b['seed']==seed);final=next(c for c in parent['results'] if c['seed']==seed and c['arm']=='adaptive_bounded')['episodes'][-1];queries,generation=generate(seed,boot,scfg,cfg,a.preflight);assert d['queries']==queries and d['query_source_hash']==identity(queries)
        for q in queries:assert independent_reference(q)==q['reference']['indices']
        for encoding in cfg['encodings']:assert d['views'][encoding]==[rename(q,encoding,seed+cfg['renaming_seed_offset']+i*1009) for i,q in enumerate(queries)]
        for k,v in generation.items():assert d['generation'][k]==v
        for name,cp in [('bootstrap',boot),('adaptive_final',final)]:
            assert sha(cp['checkpoint'])==cp['checkpoint_sha256'];loaded=torch.load(cp['checkpoint'],map_location='cpu',weights_only=False);source_adapters[(seed,name)]=state_hash(loaded['adapter'] if name=='bootstrap' else loaded['actor']['adapter'])
        datasets[seed]=d;counts=collections.Counter((q['split'],q['reference']['category']) for q in queries);inventory.append(dict(seed=seed,query_counts={split:sum(q['split']==split for q in queries) for split in ('tutorial','fresh')},categories={split:{cat:n for (sp,cat),n in counts.items() if sp==split} for split in ('tutorial','fresh')},unique_tutorial_prompt_labels=len({identity([q['prompt'],q['teacher_index']]) for q in queries if q['split']=='tutorial'}),tutorial_exposed_queries=sum(q['split']=='tutorial' and q['bootstrap_prompt_label_exposures']>0 for q in queries),generation=generation,all_floyd_reference_sets_exact=True))
    running={seed:dict(inference_calls=0,inference_tokens=0,inference_attention_elements=0,update_calls=0,update_tokens=0,update_attention_elements=0,audit_calls=0,audit_tokens=0) for seed in cfg['source_seeds']};assert len(data['results'])==len(cfg['source_seeds'])*len(cfg['states'])*len(cfg['encodings'])
    for cell in data['results']:
        seed,state,encoding=cell['seed'],cell['state'],cell['encoding'];d=datasets[seed];assert cell['dataset_sha256']==sha(a.source/f'{seed}-queries.json');assert len(cell['rows'])==len(d['queries']);assert cell['actor_weights_optimizer_memory_rng_unchanged']
        if state!='base':assert cell['adapter_sha256']==source_adapters[(seed,state)]
        for i,(r,q,view) in enumerate(zip(cell['rows'],d['queries'],d['views'][encoding])):
            assert (r['index'],r['split'],r['category'],r['reference_indices'],r['teacher_index'])==(i,q['split'],q['reference']['category'],q['reference']['indices'],q['teacher_index']);assert r['bootstrap_index_exposures']==q['bootstrap_index_exposures'] and r['bootstrap_prompt_label_exposures']==q['bootstrap_prompt_label_exposures'];assert r['mapping_utf8_bytes']==len(canonical(view['mapping']).encode())
            prompt=view['prompt']
            if prompt not in cache:
                text=tok.apply_chat_template([dict(role='system',content=RULES),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True);cache[prompt]=len(tok(text)['input_ids'])
            assert r['tokens']==cache[prompt];logits=np.asarray(r['logits'],dtype=np.float64);assert len(logits)==len(view['actions']) and np.isfinite(logits).all();prediction=int(np.argmax(logits));assert r['prediction']==prediction and r['rule_correct']==(prediction in r['reference_indices']) and r['teacher_exact']==(prediction==r['teacher_index']);shifted=logits-logits.max();p=np.exp(shifted);p/=p.sum();ce=float(np.log(np.exp(shifted).sum())-shifted[r['teacher_index']]);prob=float((.95*p+.05/len(p))[r['reference_indices']].sum());assert math.isclose(ce,r['teacher_cross_entropy'],rel_tol=5e-6,abs_tol=5e-6);assert math.isclose(prob,r['rule_probability'],rel_tol=5e-6,abs_tol=5e-6)
        work=dict(inference_calls=len(cell['rows']),inference_tokens=sum(r['tokens'] for r in cell['rows']),inference_attention_elements=336*sum(r['tokens']**2 for r in cell['rows']),update_calls=0,update_tokens=0,update_attention_elements=0,audit_calls=0,audit_tokens=0);assert cell['work']==work
        for k,v in work.items():running[seed][k]+=v;assert cell['costs'][k]==running[seed][k]
        subsets={'tutorial':[r for r in cell['rows'] if r['split']=='tutorial'],'fresh':[r for r in cell['rows'] if r['split']=='fresh']}
        for n in cfg['fresh_world_sizes']:
            subsets['fresh_n'+str(n)]=[r for r,q in zip(cell['rows'],d['queries']) if q.get('world_size')==n]
        subsets['tutorial_exposed']=[r for r in subsets['tutorial'] if r['bootstrap_prompt_label_exposures']>0];subsets['tutorial_unexposed']=[r for r in subsets['tutorial'] if r['bootstrap_prompt_label_exposures']==0]
        groups={k:summarize(v) for k,v in subsets.items() if v};categories={split:{category:summarize([r for r in subsets[split] if r['category']==category]) for category in sorted({r['category'] for r in subsets[split]})} for split in ('tutorial','fresh')};adequate=groups['tutorial']['rule_accuracy']>=.95 and all(c['rule_accuracy']>=.8 for c in categories['tutorial'].values() if c['n']>=10);allsummary.append(dict(seed=seed,state=state,encoding=encoding,groups=groups,categories=categories,tutorial_gate=adequate,query_seconds=cell['query_seconds'],work=work,cuda_peak_allocated_bytes=cell['cuda_peak_allocated_bytes'],source_adapter_sha256=cell['adapter_sha256'],all_rows_metrics_tokens_reference_source_verified=True))
    for seed in cfg['source_seeds']:
        for state in cfg['states']:
            cells=[c for c in data['results'] if c['seed']==seed and c['state']==state];assert {c['encoding'] for c in cells}==set(cfg['encodings']) and len({c['adapter_sha256'] for c in cells})==1;base=next(c for c in cells if c['encoding']=='original')
            for encoding in ('fresh_nonce','compact'):
                other=next(c for c in cells if c['encoding']==encoding)
                for split in ('tutorial','fresh'):
                    pairs0=[(x,y) for x,y in zip(base['rows'],other['rows']) if x['split']==split];pairs.append(dict(seed=seed,state=state,encoding=encoding,split=split,n=len(pairs0),prediction_changes=sum(x['prediction']!=y['prediction'] for x,y in pairs0),correct_to_wrong=sum(x['rule_correct'] and not y['rule_correct'] for x,y in pairs0),wrong_to_correct=sum(not x['rule_correct'] and y['rule_correct'] for x,y in pairs0),both_wrong=sum(not x['rule_correct'] and not y['rule_correct'] for x,y in pairs0)))
    assert sha(path)==before and sha(a.parent/'complete.json')==manifest['source_complete_sha256'];result=dict(status='E31_PREFLIGHT_AUDITED' if a.preflight else 'E31_AUDITED',source_sha256=before,source_commit=manifest['git_commit'],audit_script_sha256=sha(__file__),inventory=inventory,cells=allsummary,paired_renaming=pairs,total_queries=sum(c['work']['inference_calls'] for c in data['results']),total_tokens=sum(c['work']['inference_tokens'] for c in data['results']),total_attention_elements=sum(c['work']['inference_attention_elements'] for c in data['results']),total_query_seconds=sum(c['query_seconds'] for c in data['results']),model_load_seconds_once_per_seed=sum(next(c['model_load_seconds'] for c in data['results'] if c['seed']==s) for s in cfg['source_seeds']),audit_seconds=time.perf_counter()-start,limits='Static teacher-induced states and correlated repeated queries. All reference sets independently match Floyd-Warshall observed-map distances, metrics recompute from recorded logits, and source adapters match. Neural logits are not independently reexecuted here. Source bootstrap and model pretraining costs are not erased; query-local encodings are not a deployed memory system.');a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','total_queries','total_tokens','audit_seconds')}))
if __name__=='__main__':main()
