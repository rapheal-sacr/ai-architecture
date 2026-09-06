"""Independent E30 trajectory/state/cost reconstruction; no candidate mutation."""
from pathlib import Path
import argparse,copy,hashlib,json,math,subprocess,time
import numpy as np,torch
from transformers import AutoTokenizer
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,render_prompt,canonical,identity,RULES
from language_delivery_agent import tensor_bytes
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_cp(path,expected):
    assert sha(path)==expected
    return torch.load(path,map_location='cpu',weights_only=False)
def counts(records):
    return dict(calls=len(records),tokens=sum(x['tokens'] for x in records),attention_elements=336*sum(x['tokens']**2 for x in records))
def add_work(work,kind,records):
    for k,v in counts(records).items():work[kind+'_'+k]+=v

def audit(source,model,preflight=False):
    start=time.perf_counter();path=source/'complete.json';source_hash=sha(path);data=json.loads(path.read_text());cfg=data['manifest']['config'];assert data['status']==('PREFLIGHT_ONLY' if preflight else 'E30_COMPLETE');torch.set_num_threads(1)
    if not preflight:
        assert data['manifest']['git_commit'].startswith('028bffe')
        for f,h in data['manifest']['code_identity'].items():
            blob=subprocess.check_output(['git','show',data['manifest']['git_commit']+':research/adaptive_latent_workspace/'+f],cwd=HERE);assert hashlib.sha256(blob).hexdigest()==h
    for f,h in data['manifest']['code_identity'].items():
        if preflight and f=='protocol_e30.json':continue # documented criteria-only clarification
        assert sha(HERE/f)==h
    tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True,trust_remote_code=False);token_cache={}
    def tokens(prompt):
        if prompt not in token_cache:
            text=tokenizer.apply_chat_template([dict(role='system',content=RULES),dict(role='user',content=prompt)],tokenize=False,add_generation_prompt=True)
            token_cache[prompt]=len(tokenizer(text)['input_ids']);assert token_cache[prompt]<=cfg['max_prompt_tokens']
        return token_cache[prompt]
    boot_states={};boot_audits=[]
    for b in data['bootstrap']:
        seed=b['seed'];full=ObservationMemory();bounded=ObservationMemory(cfg['bounded_room_records']);planner=ObservedPlanner();examples=[];missions=[]
        for i in range(cfg['bootstrap_worlds']):
            world=DeliveryWorld(seed+700000+i,cfg['bootstrap_world_size'])
            for mi in range(cfg['bootstrap_missions_per_world']):
                obs=world.reset(mi%2);steps=0;done=False
                for _ in range(cfg['bootstrap_steps_per_mission']):
                    full.observe(obs);bounded.observe(obs);assert full.view(obs['world'])==bounded.view(obs['world']);action=planner.act(obs,full);prompt,actions=render_prompt(obs,bounded);examples.append(dict(prompt=prompt,actions=actions,index=actions.index(action),observation=copy.deepcopy(obs)));obs,reward=world.step(action);steps+=1
                    if reward:done=True;break
                assert done;missions.append(dict(world=world.world,target=world.target,actions=steps,success=done))
        assert examples==b['examples'] and identity(examples)==b['example_sha256'] and missions==b['teacher_missions'];assert len(examples)==b['teacher_actions']
        trainrng=np.random.default_rng(seed+800000);selected=[dict(tokens=tokens(examples[int(trainrng.integers(len(examples)))]['prompt'])) for _ in range(cfg['bootstrap_updates'])];work=counts(selected)
        assert len(b['training_trace'])==cfg['bootstrap_updates'];assert all(math.isfinite(z[k]) for z in b['training_trace'] for k in ('loss','grad_norm'))
        for k,v in work.items():assert b['work']['update_'+k]==v
        cp=load_cp(b['checkpoint'],b['checkpoint_sha256']);boot_states[seed]=cp
        assert cp['replay']==[] and cp['seen']==0;assert cp['work']['update_calls']==cfg['bootstrap_updates']
        assert {int(s['step']) for s in cp['optimizer']['state'].values()}=={cfg['bootstrap_updates']};assert tensor_bytes(cp['adapter'])==b['work']['adapter_parameter_bytes'];assert tensor_bytes(cp['optimizer'])==b['work']['optimizer_tensor_bytes']
        boot_audits.append(dict(seed=seed,teacher_missions=len(missions),teacher_actions=len(examples),teaching_and_work_reconstructed=True,checkpoint_sha256=b['checkpoint_sha256']))
    assert len(data['results'])==len(cfg['seeds'])*len(cfg['arms']);rows=[];checkpoints=0
    for case in data['results']:
        seed=case['seed'];arm=case['arm'];adaptive=arm=='adaptive_bounded';neural=arm!='observed_planner';boot=boot_states[seed];mem=ObservationMemory(None if arm in ('frozen_full','observed_planner') else cfg['bounded_room_records']);worlds=[DeliveryWorld(seed+offset,n) for offset,n in zip(cfg['world_seed_offsets'],cfg['world_sizes'])];planner=ObservedPlanner();replay=[];seen=0;lrng=np.random.default_rng(seed+600000);arng=np.random.default_rng(seed+500000);work=dict(inference_calls=0,inference_tokens=0,inference_attention_elements=0,update_calls=0,update_tokens=0,update_attention_elements=0,audit_calls=0,audit_tokens=0);changed=0;last_adapter=state_hash(boot['adapter']);old_steps=0
        assert len(case['episodes'])==len(cfg['missions']);step_waste=0;local_counts=dict(delivery_opportunities=0,missed_delivery=0,target_pickup_opportunities=0,missed_target_pickup=0,wrong_object_pickup=0,discarded_target=0,immediate_home_opportunities=0,missed_immediate_home=0)
        for mi,(ep,spec) in enumerate(zip(case['episodes'],cfg['missions'])):
            world=worlds[spec['world']]
            if spec.get('relocate'):
                old=world.shelves[world.objects[spec['target']]];world.relocate(spec['target'],next(r for r in world.rooms if r not in (world.home,old)))
            obs=world.reset(spec['target']);definition=world.audit_state();definition={k:definition[k] for k in ('world','home','edges','objects','shelves')};assert identity(definition)==ep['world_definition_hash'];assert obs==ep['initial_observation'];trajectory=ep['trajectory'];assert len(trajectory)==(ep['actions'] if neural else 0);reward=0
            for j,event in enumerate(ep['events']):
                assert not reward and obs==event['observation'];mem.observe(obs);prompt,actions=render_prompt(obs,mem);assert event['action'] in actions
                if neural:
                    record=trajectory[j];assert record['prompt']==prompt and record['actions']==actions and actions[record['index']]==event['action'];assert tokens(prompt)==record['tokens'];assert math.isfinite(record['behavior_logp']) and record['behavior_logp']<=0
                    # np.choice(p=...) consumes one uniform draw regardless of probabilities.
                    # This checks RNG progression, not unrecorded alternate-action logits.
                    arng.choice(len(actions),p=np.full(len(actions),1/len(actions)))
                else:assert event['action']==planner.act(obs,mem)
                action=event['action']
                if any(x.startswith('deliver:') for x in actions):
                    local_counts['delivery_opportunities']+=1;local_counts['missed_delivery']+=int(not action.startswith('deliver:'))
                if 'pick:'+obs['target'] in actions:
                    local_counts['target_pickup_opportunities']+=1;local_counts['missed_target_pickup']+=int(action!='pick:'+obs['target'])
                local_counts['wrong_object_pickup']+=int(action.startswith('pick:') and action!='pick:'+obs['target'])
                local_counts['discarded_target']+=int(action.startswith('drop:') and obs['carrying']==obs['target'])
                if obs['carrying']==obs['target'] and obs['home'] in obs['exits']:
                    local_counts['immediate_home_opportunities']+=1;local_counts['missed_immediate_home']+=int(action!='move:'+obs['home'])
                if action.startswith(('wait:','drop:')):step_waste+=1
                obs,reward=world.step(event['action']);assert obs==event['next_observation'] and reward==event['reward']
            assert ep['actions']==len(ep['events']) and 0<ep['actions']<=spec['budget'];assert bool(reward)==ep['success'];assert ep['return_value']==float(reward)-cfg['reward_step_penalty']*ep['actions'];assert reward or ep['actions']==spec['budget']
            assert ep['memory_bytes']==mem.bytes() and ep['memory_records']==len(mem.records) and ep['evictions']==mem.evictions;assert ep['temporary_trajectory_utf8_bytes']==len(canonical(trajectory).encode());add_work(work,'inference',trajectory)
            selected=[]
            if adaptive:
                current=[]
                for item in trajectory:
                    record=copy.deepcopy(item);record['advantage']=ep['return_value']-cfg['return_baseline'];current.append(record);seen+=1
                    if len(replay)<cfg['experience_capacity']:replay.append(record)
                    else:
                        idx=int(lrng.integers(seen))
                        if idx<cfg['experience_capacity']:replay[idx]=record
                for i in range(cfg['online_updates_per_mission']):
                    pool=current if i<cfg['online_updates_per_mission']//2 else replay;selected.append(pool[int(lrng.integers(len(pool)))])
            assert len(selected)==len(ep['updates']);assert all(math.isfinite(z[k]) for z in ep['updates'] for k in ('loss','grad_norm'));add_work(work,'update',selected)
            cp=load_cp(ep['checkpoint'],ep['checkpoint_sha256']);checkpoints+=1;assert cp['config']==cfg and cp['memory']==mem.state() and cp['world_definition_hash']==ep['world_definition_hash'];assert (cp['seed'],cp['arm'],cp['mission'])==(seed,arm,mi)
            if neural:
                actor=cp['actor'];assert actor['replay']==replay and actor['seen']==seen;assert actor['learning_rng']==lrng.bit_generator.state and actor['action_rng']==arng.bit_generator.state;assert actor['work']==work;assert torch.equal(actor['torch_rng'],boot['torch_rng']);assert all(torch.equal(a,b) for a,b in zip(actor['cuda_rng'],boot['cuda_rng']))
                adapter_hash=state_hash(actor['adapter']);changed+=int(adapter_hash!=last_adapter);last_adapter=adapter_hash
                if adaptive:
                    assert {int(s['step']) for s in actor['optimizer']['state'].values()}=={cfg['bootstrap_updates']+(mi+1)*cfg['online_updates_per_mission']};assert all(g['lr']==cfg['online_learning_rate'] for g in actor['optimizer']['param_groups'])
                else:assert actor['optimizer'] is None and adapter_hash==state_hash(boot['adapter'])
            else:assert cp['actor'] is None
        assert case['final_memory_state']==mem.state() and case['final_memory_bytes']==mem.bytes();assert case['total_actions']==sum(e['actions'] for e in case['episodes']) and case['total_successes']==sum(e['success'] for e in case['episodes'])
        state_cost=dict(room_utf8_bytes=mem.bytes(),numpy_rng_utf8_bytes=0)
        if neural:
            assert case['initial_adapter_hash']==state_hash(boot['adapter']);assert case['audit']['base_unchanged'] and case['audit']['complete_actor_restoration_exact'] and case['audit']['restored_logits_exact']
            for k,v in work.items():
                if not k.startswith('audit'):assert case['costs'][k]==v
            assert case['costs']['audit_calls']==2;assert case['costs']['audit_tokens']==2*trajectory[-1]['tokens']
            assert case['costs']['adapter_parameter_bytes']==tensor_bytes(actor['adapter']);assert case['costs']['optimizer_tensor_bytes']==tensor_bytes(actor['optimizer']);assert case['costs']['replay_utf8_bytes']==len(canonical(replay).encode());assert case['costs']['replay_records']==len(replay)
            state_cost.update(numpy_rng_utf8_bytes=len(canonical(dict(action=actor['action_rng'],learning=actor['learning_rng'])).encode()),**case['costs']);state_cost['persistent_parameter_optimizer_rng_replay_room_bytes']=sum(state_cost[k] for k in ('base_parameter_bytes','adapter_parameter_bytes','optimizer_tensor_bytes','rng_tensor_bytes','numpy_rng_utf8_bytes','replay_utf8_bytes','room_utf8_bytes'))
        eps=case['episodes'];suc=lambda a,b:sum(e['success'] for e in eps[a:b]);actions=lambda a,b:sum(e['actions'] for e in eps[a:b]);acquired=(suc(0,4)>=3) if not preflight else None
        row=dict(seed=seed,arm=arm,total_successes=case['total_successes'],total_actions=case['total_actions'],wait_or_drop_actions=step_waste,posthoc_local_action_diagnostics=local_counts,adapter_changed_missions=changed,costs=state_cost,setup_seconds=case['setup_seconds'],loop_seconds=case['loop_seconds'],acting_seconds=sum(e['acting_seconds'] for e in eps),update_seconds=sum(e['update_seconds'] for e in eps),cuda_peak_allocated_bytes=case['cuda_peak_allocated_bytes'],max_temporary_trajectory_utf8_bytes=max(e['temporary_trajectory_utf8_bytes'] for e in eps),mission_actions=[e['actions'] for e in eps],mission_successes=[e['success'] for e in eps],trajectory_memory_replay_work_checkpoint_audit=True)
        if not preflight:row.update(initial_successes=suc(0,4),initial_actions=actions(0,4),acquired=acquired,larger_successes=suc(4,8),larger_actions=actions(4,8),larger_competent=suc(4,8)>=3,return_successes=suc(8,10),return_actions=actions(8,10),return_retention_interpretable=acquired,relocation_success=eps[10]['success'],relocation_actions=eps[10]['actions'],other_object_success=eps[11]['success'])
        rows.append(row)
    for seed in cfg['seeds']:
        group=[c for c in data['results'] if c['seed']==seed];assert len(group)==len(cfg['arms']) and {c['arm'] for c in group}==set(cfg['arms']);assert len({c['initial_adapter_hash'] for c in group if c['arm']!='observed_planner'})==1
        for mi in range(len(cfg['missions'])):assert len({c['episodes'][mi]['world_definition_hash'] for c in group})==1 and len({identity(c['episodes'][mi]['initial_observation']) for c in group})==1
    assert sha(path)==source_hash
    return dict(status='E30_PREFLIGHT_AUDITED' if preflight else 'E30_AUDITED',source_sha256=source_hash,source_commit=data['manifest']['git_commit'],audit_script_sha256=sha(__file__),bootstrap_audits=boot_audits,mission_checkpoints_audited=checkpoints,rows=rows,audit_seconds=time.perf_counter()-start,tokenized_distinct_prompts=len(token_cache),limits='Trajectory and state reconstruction includes actual chosen actions, memory, reservoir, private RNG progression, update selections and token counts. It does not recompute alternate-action logits or neural gradients; scorer supplied base-hash and final restore checks. Full Python overhead, base buffers, energy and pretraining cost are not measured. Attention elements are forward attention sizes, not FLOPs. Bootstrap and audit work are separate from operational use.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();result=audit(a.source,a.model,a.preflight);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],checkpoints=result['mission_checkpoints_audited'],seconds=result['audit_seconds'])))
if __name__=='__main__':main()
