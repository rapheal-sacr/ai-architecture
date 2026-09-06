"""Independent E32 reconstruction; optional partial audit never denotes completion."""
from pathlib import Path
import argparse,collections,hashlib,json,math,subprocess,time
import numpy as np,torch
from transformers import AutoTokenizer
from delivery_training_data import make_data,assert_disjoint
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,render_prompt,canonical,identity,RULES
from language_delivery_agent import tensor_bytes
from record_e31 import independent_reference,summarize
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def count_work(tokens):return dict(inference_calls=len(tokens),inference_tokens=sum(tokens),inference_attention_elements=336*sum(n*n for n in tokens),update_calls=0,update_tokens=0,update_attention_elements=0,audit_calls=0,audit_tokens=0)

def audit_loop(result,data,cfg,tokens,neural):
    specs=sorted((w for w in data['worlds'] if w['domain']=='evaluation'),key=lambda w:w['index']);worlds=[DeliveryWorld(w['seed'],w['rooms']) for w in specs];mem=ObservationMemory();planner=ObservedPlanner();rng=np.random.default_rng(data['seed']+500000);counts=[];assert len(result['episodes'])==len(cfg['eval_missions'])
    for mi,(ep,spec) in enumerate(zip(result['episodes'],cfg['eval_missions'])):
        world=worlds[spec['world']]
        if spec.get('relocate'):
            old=world.shelves[world.objects[spec['target']]];world.relocate(spec['target'],next(r for r in world.rooms if r not in (world.home,old)))
        obs=world.reset(spec['target']);state=world.audit_state();definition={k:state[k] for k in ('world','home','edges','objects','shelves')};assert ep['mission']==mi and ep['initial_observation']==obs and ep['world_definition_hash']==identity(definition);reward=0;assert len(ep['trajectory'])==(ep['actions'] if neural else 0)
        for j,event in enumerate(ep['events']):
            assert not reward and event['observation']==obs;mem.observe(obs);prompt,actions=render_prompt(obs,mem);action=event['action'];assert action in actions
            if neural:
                r=ep['trajectory'][j];assert (r['prompt'],r['actions'])==(prompt,actions) and actions[r['index']]==action;assert r['tokens']==tokens(prompt) and math.isfinite(r['behavior_logp']) and r['behavior_logp']<=0;counts.append(r['tokens']);rng.choice(len(actions),p=np.full(len(actions),1/len(actions)))
            else:assert action==planner.act(obs,mem)
            obs,reward=world.step(action);assert obs==event['next_observation'] and reward==event['reward']
        assert bool(reward)==ep['success'] and 0<ep['actions']==len(ep['events'])<=spec['budget'];assert reward or ep['actions']==spec['budget'];assert ep['return_value']==float(reward)-.01*ep['actions'];assert ep['memory_state']==mem.state() and ep['memory_bytes']==mem.bytes();assert ep['trajectory_utf8_bytes']==len(canonical(ep['trajectory']).encode());assert ep['action_rng']==(rng.bit_generator.state if neural else None)
    assert result['total_successes']==sum(e['success'] for e in result['episodes']) and result['total_actions']==sum(e['actions'] for e in result['episodes']);assert result['final_memory_state']==mem.state() and result['final_memory_bytes']==mem.bytes();assert result['work']==(count_work(counts) if neural else {})
    return dict(actual_trajectories_memory_rng_tokens_exact=True,actions=result['total_actions'],successes=result['total_successes'],work=result['work'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');p.add_argument('--partial',action='store_true');a=p.parse_args();start=time.perf_counter();torch.set_num_threads(1);manifest=json.loads((a.source/'manifest.json').read_text());cfg=manifest['config'];scfg=manifest['source_config'];assert manifest['preflight']==a.preflight
    if not a.preflight:assert manifest['git_commit'].startswith('7d4edc9')
    for f,h in manifest['code_identity'].items():
        assert sha(HERE/f)==h
        if not a.preflight:assert hashlib.sha256(subprocess.check_output(['git','show',manifest['git_commit']+':research/adaptive_latent_workspace/'+f],cwd=HERE)).hexdigest()==h
    if a.partial:
        dm=json.loads((a.source/'data_manifest.json').read_text());cases=[];planners=[]
        for seed in cfg['seeds']:
            pp=a.source/f'{seed}-planner.json'
            if pp.exists():planners.append(json.loads(pp.read_text()))
            for arm in cfg['arms']:
                fp=a.source/f'{seed}-{arm["name"]}.json'
                if fp.exists():cases.append(json.loads(fp.read_text()))
        source_hash=None
    else:
        cp=a.source/'complete.json';source_hash=sha(cp);complete=json.loads(cp.read_text());assert complete['status']==('PREFLIGHT_ONLY' if a.preflight else 'E32_COMPLETE');assert complete['manifest']==manifest;dm=dict(files=complete['dataset_files'],disjoint=complete['data_disjoint']);cases=complete['results'];planners=complete['planners'];assert len(cases)==len(cfg['seeds'])*len(cfg['arms']) and len(planners)==len(cfg['seeds'])
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True,trust_remote_code=False);cache={}
    def tokens(prompt):
        if prompt not in cache:
            text=tokenizer.apply_chat_template([dict(role='system',content=RULES),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True);cache[prompt]=len(tokenizer(text)['input_ids']);assert cache[prompt]<=scfg['max_prompt_tokens']
        return cache[prompt]
    datasets={}
    for ds in dm['files']:
        assert sha(ds['path'])==ds['sha256'];data=json.loads(Path(ds['path']).read_text());assert data==make_data(data['seed'],cfg)
        for q in data['queries']:assert independent_reference(q)==q['reference']['indices']
        datasets[data['seed']]=data
    assert assert_disjoint(list(datasets.values()))==dm['disjoint'];planner_audits=[dict(seed=r['seed'],**audit_loop(r,datasets[r['seed']],cfg,tokens,False)) for r in planners];rows=[];checkpoints=0;pair_state={}
    for case in cases:
        seed=case['seed'];arm=case['arm'];data=datasets[seed];assert arm in cfg['arms'];assert case['data_hash']==data['example_sha256'] and case['order_hash']==data['order_sha256'];assert case['teacher_actions']==data['teacher_actions'] and case['base_unchanged'];assert len(case['trace'])==max(cfg['milestones'])//arm['batch'];assert len(case['stages'])==len(cfg['milestones'])
        for i,tr in enumerate(case['trace']):
            assert tr['offset']==i*arm['batch'] and tr['indices']==data['order'][tr['offset']:tr['offset']+arm['batch']];assert tr['examples']==len(tr['example_losses'])==arm['batch'] and tr['optimizer_step']==i+1;assert all(math.isfinite(v) and v>=0 for v in tr['example_losses']);assert math.isclose(tr['loss'],float(np.mean(tr['example_losses'])),abs_tol=1e-12,rel_tol=1e-12) and math.isfinite(tr['grad_norm'])
        for stage,milestone in zip(case['stages'],cfg['milestones']):
            assert stage['milestone']==milestone and stage['optimizer_steps']==milestone//arm['batch'];assert stage['evaluation_did_not_update_parameters'] and stage['post_evaluation_training_state_restore_exact'];assert sha(stage['checkpoint'])==stage['checkpoint_sha256'];cp=torch.load(stage['checkpoint'],map_location='cpu',weights_only=False);checkpoints+=1;actor=cp['actor'];assert state_hash(actor)==stage['state_sha256'];assert cp['seed']==seed and cp['arm']==arm and cp['milestone']==milestone and cp['config']==cfg and cp['source_config']==scfg;assert cp['data_hash']==data['example_sha256'] and cp['order_hash']==data['order_sha256']
            assert actor['replay']==[] and actor['seen']==0 and actor['action_rng']==np.random.default_rng(seed+500000).bit_generator.state and actor['learning_rng']==np.random.default_rng(seed+600000).bit_generator.state
            assert {int(s['step']) for s in actor['optimizer']['state'].values()}=={milestone//arm['batch']};assert all(g['lr']==arm['learning_rate'] for g in actor['optimizer']['param_groups']);assert all(torch.isfinite(v).all() for s in actor['optimizer']['state'].values() for v in s.values() if torch.is_tensor(v))
            lengths=[tokens(data['examples'][i]['prompt']) for i in data['order'][:milestone]];work=dict(inference_calls=0,inference_tokens=0,inference_attention_elements=0,update_calls=milestone,update_tokens=sum(lengths),update_attention_elements=336*sum(n*n for n in lengths),audit_calls=0,audit_tokens=0);assert actor['work']==stage['training_work']==work
            costs=stage['training_costs'];assert costs['adapter_parameter_bytes']==tensor_bytes(actor['adapter']) and costs['optimizer_tensor_bytes']==tensor_bytes(actor['optimizer']);assert costs['replay_utf8_bytes']==len(canonical([]).encode()) and costs['replay_records']==0
            for k,v in work.items():assert costs[k]==v
            key=seed;shape_state=state_hash(dict(torch_rng=actor['torch_rng'],cuda_rng=actor['cuda_rng']))
            if key in pair_state:assert pair_state[key]==shape_state
            else:pair_state[key]=shape_state
            exposure=np.bincount(data['order'][:milestone],minlength=len(data['queries']));pooled={}
            for q,n in zip(data['queries'],exposure):
                key=identity([q['prompt'],q['teacher_index']]);pooled[key]=pooled.get(key,0)+int(n)
            diag=stage['diagnostic'];assert len(diag['rows'])==len(data['queries'])
            for i,(r,q) in enumerate(zip(diag['rows'],data['queries'])):
                assert (r['index'],r['category'],r['reference_indices'],r['teacher_index'])==(i,q['reference']['category'],q['reference']['indices'],q['teacher_index']);assert r['tokens']==tokens(q['prompt']);assert r['index_exposures']==int(exposure[i]) and r['prompt_label_exposures']==pooled[identity([q['prompt'],q['teacher_index']])];logits=np.asarray(r['logits'],dtype=np.float64);assert len(logits)==len(q['actions']) and np.isfinite(logits).all();pred=int(np.argmax(logits));assert r['prediction']==pred and r['rule_correct']==(pred in r['reference_indices']) and r['teacher_exact']==(pred==r['teacher_index']);shift=logits-logits.max();soft=np.exp(shift);soft/=soft.sum();ce=float(np.log(np.exp(shift).sum())-shift[r['teacher_index']]);pr=float((.95*soft+.05/len(soft))[r['reference_indices']].sum());assert math.isclose(ce,r['teacher_cross_entropy'],rel_tol=5e-6,abs_tol=5e-6) and math.isclose(pr,r['rule_probability'],rel_tol=5e-6,abs_tol=5e-6)
            assert diag['work']==count_work([r['tokens'] for r in diag['rows']]);summary=summarize(diag['rows']);cats={k:summarize([r for r in diag['rows'] if r['category']==k]) for k in sorted({r['category'] for r in diag['rows']})};mastered=summary['rule_accuracy']>=.95 and all(c['rule_accuracy']>=.8 for c in cats.values() if c['n']>=10);op=stage['closed_loop'];loop=audit_loop(op,data,cfg,tokens,True);ep=op['episodes'];success=lambda start,end:sum(e['success'] for e in ep[start:end]);row=dict(seed=seed,arm=arm['name'],milestone=milestone,tutorial=summary,categories=cats,tutorial_mastered=mastered,tutorial_exposed=sum(r['prompt_label_exposures']>0 for r in diag['rows']),teacher_actions=data['teacher_actions'],training_seconds=stage['cumulative_training_seconds'],training_work=work,optimizer_steps=stage['optimizer_steps'],training_costs=costs,diagnostic_seconds=diag['seconds'],diagnostic_work=diag['work'],closed_loop=loop,acting_seconds=op['acting_seconds'],mission_successes=[e['success'] for e in ep],mission_actions=[e['actions'] for e in ep],final_memory_bytes=op['final_memory_bytes'],training_cuda_peak_bytes=stage['training_cuda_peak_bytes'],evaluation_cuda_peak_bytes=stage['evaluation_cuda_peak_bytes'],checkpoint_sha256=stage['checkpoint_sha256'],checkpoint_optimizer_rng_work_verified=True)
            if not a.preflight:row.update(initial_successes=success(0,4),initial_acquired=success(0,4)>=3,larger_successes=success(4,8),larger_competent=success(4,8)>=3,return_successes=success(8,10),return_retention_interpretable=success(0,4)>=3,relocation_success=ep[10]['success'],other_success=ep[11]['success'])
            rows.append(row)
    groups=[]
    for seed in cfg['seeds']:
        selected=[c for c in cases if c['seed']==seed]
        if not selected:continue
        assert len({c['initial_adapter_sha256'] for c in selected})==1;assert len({c['trace'][0]['example_losses'][0] for c in selected})==1
        for milestone in cfg['milestones']:
            entries=[r for r in rows if r['seed']==seed and r['milestone']==milestone];assert len({identity(r['training_work']) for r in entries})==1;groups.append(dict(seed=seed,milestone=milestone,arms_audited=len(entries),all_four_arms_present=len(entries)==len(cfg['arms']),same_initial_weights_first_loss_and_gradient_example_work=True))
    result=dict(status='E32_PARTIAL_AUDITED' if a.partial else 'E32_PREFLIGHT_AUDITED' if a.preflight else 'E32_AUDITED',source_commit=manifest['git_commit'],source_sha256=source_hash,audit_script_sha256=sha(__file__),completed_cases_audited=len(cases),expected_cases=len(cfg['seeds'])*len(cfg['arms']),checkpoints_audited=checkpoints,expected_checkpoints=len(cfg['seeds'])*len(cfg['arms'])*len(cfg['milestones']),data_disjoint=dm['disjoint'],planner_audits=planner_audits,paired_work=groups,rows=rows,audit_seconds=time.perf_counter()-start,limits='Every completed checkpoint, recorded group/index order, work count, reference/metric and real trajectory is reconstructed. Scored neural gradients are not reexecuted here; historical batch1 and independent joint-gradient preflights cover training implementation. Milestone training work is cumulative and must not be double-counted. Evaluations are charged separately and never enter training. This is a conventional baseline study, not RSI or the full architecture.');a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','completed_cases_audited','checkpoints_audited','audit_seconds')}))
if __name__=='__main__':main()
