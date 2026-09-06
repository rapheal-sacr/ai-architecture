"""E30 integrated but bounded language/memory/feedback pilot; not full RSI."""
from pathlib import Path
import argparse,copy,hashlib,json,subprocess,time,gc
import numpy as np,torch
import transformers,importlib.metadata
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,render_prompt,identity,canonical
from language_delivery_agent import LanguageActor
from preflight_language_adapter import base_hash,adapter_state
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def bootstrap(seed,cfg,model,out):
    start=time.perf_counter();teacher=ObservedPlanner();full=ObservationMemory();bounded=ObservationMemory(cfg['bounded_room_records']);examples=[];missions=[]
    for i in range(cfg['bootstrap_worlds']):
        world=DeliveryWorld(seed+700000+i,cfg['bootstrap_world_size'])
        for mission in range(cfg['bootstrap_missions_per_world']):
            obs=world.reset(mission%2);trajectory=[];success=False
            for step in range(cfg['bootstrap_steps_per_mission']):
                full.observe(obs);bounded.observe(obs);assert full.view(obs['world'])==bounded.view(obs['world']);action=teacher.act(obs,full);prompt,actions=render_prompt(obs,bounded);trajectory.append(dict(prompt=prompt,actions=actions,index=actions.index(action),observation=copy.deepcopy(obs)));obs,reward=world.step(action)
                if reward:success=True;break
            assert success;examples+=trajectory;missions.append(dict(world=world.world,target=world.target,actions=len(trajectory),success=success))
    generation_seconds=time.perf_counter()-start;actor=LanguageActor(model,cfg,seed);base=base_hash(actor.model);actor.start_optimizer(cfg['bootstrap_learning_rate']);rng=np.random.default_rng(seed+800000);trace=[];torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(cfg['bootstrap_updates']):trace.append(actor.supervised_update(examples[int(rng.integers(len(examples)))]))
    torch.cuda.synchronize();train_seconds=time.perf_counter()-start;assert base_hash(actor.model)==base;state=actor.state();cp=out/f'{seed}-bootstrap.pt';torch.save(state,cp);result=dict(seed=seed,teacher_missions=missions,teacher_actions=sum(m['actions'] for m in missions),examples=examples,example_sha256=identity(examples),generation_seconds=generation_seconds,model_load_seconds=actor.load_seconds,training_seconds=train_seconds,training_trace=trace,work=actor.costs(),base_hash=base,checkpoint=str(cp),checkpoint_sha256=sha(cp));(out/f'{seed}-bootstrap.json').write_text(json.dumps(result,indent=2)+'\n');del actor;gc.collect();torch.cuda.empty_cache();print(json.dumps(dict(event='E30_BOOTSTRAP_COMPLETE',seed=seed,teacher_actions=result['teacher_actions'],training_seconds=train_seconds)),flush=True);return result

def run_case(seed,arm,cfg,model,source,out):
    start=time.perf_counter();memory=ObservationMemory(None if arm in ('observed_planner','frozen_full') else cfg['bounded_room_records']);planner=ObservedPlanner() if arm=='observed_planner' else None;actor=None;base=None;initial_adapter=None
    if planner is None:
        actor=LanguageActor(model,cfg,seed);saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=False);actor.load_adapter(saved['adapter']);initial_adapter=state_hash(adapter_state(actor.model));base=base_hash(actor.model);assert base==source['base_hash']
        if arm=='adaptive_bounded':actor.start_optimizer(cfg['online_learning_rate'],saved['optimizer'])
    setup_seconds=time.perf_counter()-start;worlds=[DeliveryWorld(seed+offset,n) for offset,n in zip(cfg['world_seed_offsets'],cfg['world_sizes'])];episodes=[];max_cuda=0;loop_start=time.perf_counter()
    for mi,spec in enumerate(cfg['missions']):
        world=worlds[spec['world']]
        if spec.get('relocate'):
            old=world.shelves[world.objects[spec['target']]];world.relocate(spec['target'],next(r for r in world.rooms if r not in (world.home,old)))
        obs=world.reset(spec['target']);initial_observation=copy.deepcopy(obs);definition=world.audit_state();definition={k:definition[k] for k in ('world','home','edges','objects','shelves')};world_hash=identity(definition);trajectory=[];events=[];success=False;start=time.perf_counter()
        if actor:torch.cuda.reset_peak_memory_stats()
        for step in range(spec['budget']):
            memory.observe(obs)
            if planner:action=planner.act(obs,memory);record=None
            else:action,record=actor.act(obs,memory);trajectory.append(record)
            before=copy.deepcopy(obs);obs,reward=world.step(action);events.append(dict(observation=before,action=action,next_observation=copy.deepcopy(obs),reward=reward))
            if reward:success=True;break
        acting_seconds=time.perf_counter()-start;start=time.perf_counter();updates=[]
        if arm=='adaptive_bounded':updates=actor.observe_return(trajectory,success)
        if actor:torch.cuda.synchronize();max_cuda=max(max_cuda,torch.cuda.max_memory_allocated())
        update_seconds=time.perf_counter()-start;state=actor.state() if actor else None;cp=out/f'{seed}-{arm}-mission{mi}.pt';torch.save(dict(seed=seed,arm=arm,mission=mi,actor=state,memory=memory.state(),world_definition_hash=world_hash,config=cfg),cp)
        episode=dict(mission=mi,world_index=spec['world'],target_index=spec['target'],world_definition_hash=world_hash,initial_observation=initial_observation,success=success,actions=len(events),return_value=float(success)-cfg['reward_step_penalty']*len(events),acting_seconds=acting_seconds,update_seconds=update_seconds,updates=updates,temporary_trajectory_utf8_bytes=len(canonical(trajectory).encode()),memory_bytes=memory.bytes(),memory_records=len(memory.records),evictions=memory.evictions,events=events,trajectory=trajectory,checkpoint=str(cp),checkpoint_sha256=sha(cp));episodes.append(episode);print(json.dumps(dict(event='E30_MISSION',seed=seed,arm=arm,mission=mi,success=success,actions=len(events),updates=len(updates))),flush=True)
    loop_seconds=time.perf_counter()-loop_start;audit={};costs={}
    if actor:
        assert base_hash(actor.model)==base;reference=episodes[-1]['trajectory'][-1]
        with torch.no_grad():expected=actor.logits(reference['prompt'],reference['actions'],'audit')[0].clone()
        saved=torch.load(episodes[-1]['checkpoint'],map_location='cpu',weights_only=False)['actor'];expected_state=state_hash(saved)
        with torch.no_grad():
            for p in actor.params:p.zero_()
        actor.replay=[];actor.seen=-1;actor.load_state(saved);assert state_hash(actor.state())==expected_state
        with torch.no_grad():got,tokens=actor.logits(reference['prompt'],reference['actions'],'audit')
        assert torch.equal(expected,got);unchanged=initial_adapter==state_hash(adapter_state(actor.model))
        if arm.startswith('frozen'):assert unchanged
        audit=dict(base_unchanged=True,complete_actor_restoration_exact=True,restored_logits_exact=True,frozen_adapter_unchanged=unchanged,restore_audit_queries=2,restore_audit_tokens=2*tokens);costs=actor.costs();costs['audit_calls']=2;costs['audit_tokens']=2*tokens
    result=dict(seed=seed,arm=arm,setup_seconds=setup_seconds,loop_seconds=loop_seconds,bootstrap_checkpoint_sha256=source['checkpoint_sha256'] if actor else None,inherited_bootstrap=dict(teacher_actions=source['teacher_actions'],training_seconds=source['training_seconds'],work=source['work']) if actor else None,initial_adapter_hash=initial_adapter,episodes=episodes,total_successes=sum(e['success'] for e in episodes),total_actions=sum(e['actions'] for e in episodes),final_memory_bytes=memory.bytes(),final_memory_state=memory.state(),costs=costs,cuda_peak_allocated_bytes=max_cuda,audit=audit)
    (out/f'{seed}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n');del actor;gc.collect();torch.cuda.empty_cache();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args();cfg=json.loads((HERE/'protocol_e30.json').read_text());torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
    if args.preflight:cfg.update(seeds=[1711],world_sizes=[2,3],bootstrap_worlds=1,bootstrap_world_size=3,bootstrap_missions_per_world=2,bootstrap_updates=4,online_updates_per_mission=2,experience_capacity=8,missions=[dict(world=0,target=0,budget=16),dict(world=1,target=1,budget=16)])
    source=json.loads((args.model/'source_identity.json').read_text());assert source['commit']==cfg['base_model_commit'] and source['weights_sha256']==cfg['base_weights_sha256'];assert sha(args.model/'model.safetensors')==cfg['base_weights_sha256'];args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'complete.json').exists()
    transformers_repo=Path(transformers.__file__).resolve().parents[2];transformers_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=transformers_repo,text=True).strip();assert transformers_head==cfg['transformers_commit']
    files=['e28_depth.py','protocol_e30.json','e30_integrated_delivery.py','language_delivery_agent.py','delivery_world.py','preflight_language_adapter.py'];manifest=dict(config=cfg,preflight=args.preflight,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_identity={f:sha(HERE/f) for f in files},model_source=source,transformers_commit=transformers_head,versions={m:importlib.metadata.version(m) for m in ['torch','numpy','transformers','tokenizers','safetensors','huggingface-hub']},torch=torch.__version__,device=torch.cuda.get_device_name());(args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');sources=[];results=[]
    for seed in cfg['seeds']:
        source=bootstrap(seed,cfg,args.model,args.out);sources.append(source)
        for arm in cfg['arms']:results.append(run_case(seed,arm,cfg,args.model,source,args.out))
    for seed in cfg['seeds']:
        group=[x for x in results if x['seed']==seed];assert len({x['initial_adapter_hash'] for x in group if x['arm']!='observed_planner'})==1
        for mi in range(len(cfg['missions'])):assert len({x['episodes'][mi]['world_definition_hash'] for x in group})==1 and len({identity(x['episodes'][mi]['initial_observation']) for x in group})==1
    if args.preflight:
        full=next(x for x in results if x['arm']=='frozen_full');bounded=next(x for x in results if x['arm']=='frozen_bounded');assert [e['events'] for e in full['episodes']]==[e['events'] for e in bounded['episodes']]
        adaptive=next(x for x in results if x['arm']=='adaptive_bounded');assert not adaptive['audit']['frozen_adapter_unchanged']
    complete=dict(status='PREFLIGHT_ONLY' if args.preflight else 'E30_COMPLETE',manifest=manifest,bootstrap=sources,results=results,exogenous_world_pairing_verified=True);(args.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],cases=len(results),audit='passed')),flush=True)
if __name__=='__main__':main()
