"""Charged frozen-checkpoint diagnostics and real closed-loop laboratory tests."""
import copy,time
import numpy as np,torch
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,identity,canonical

def work_delta(after,before):return {k:after[k]-before[k] for k in after}
def tutorial_queries(actor,data):
    before=copy.deepcopy(actor.work);torch.cuda.synchronize();start=time.perf_counter();rows=[];exposure=np.bincount(data['presented_order'],minlength=len(data['queries']));pooled={}
    for q,n in zip(data['queries'],exposure):
        key=identity([q['prompt'],q['teacher_index']]);pooled[key]=pooled.get(key,0)+int(n)
    for i,q in enumerate(data['queries']):
        with torch.no_grad():
            logits,tokens=actor.logits(q['prompt'],q['actions'],'inference');pred=int(torch.argmax(logits));allowed=q['reference']['indices'];probs=.95*torch.softmax(logits,-1)+.05/len(q['actions']);ce=-torch.log_softmax(logits,-1)[q['teacher_index']]
        rows.append(dict(index=i,category=q['reference']['category'],reference_indices=allowed,teacher_index=q['teacher_index'],prediction=pred,rule_correct=pred in allowed,teacher_exact=pred==q['teacher_index'],teacher_cross_entropy=float(ce),rule_probability=float(probs[allowed].sum()),tokens=tokens,logits=logits.cpu().tolist(),index_exposures=int(exposure[i]),prompt_label_exposures=pooled[identity([q['prompt'],q['teacher_index']])]))
    torch.cuda.synchronize();return dict(rows=rows,seconds=time.perf_counter()-start,work=work_delta(actor.work,before))

def closed_loop(actor,data,cfg):
    memory=ObservationMemory();planner=ObservedPlanner() if actor is None else None;specs=sorted((w for w in data['worlds'] if w['domain']=='evaluation'),key=lambda w:w['index']);worlds=[DeliveryWorld(w['seed'],w['rooms']) for w in specs];episodes=[];start_all=time.perf_counter();before=copy.deepcopy(actor.work) if actor else None
    for mi,spec in enumerate(cfg['eval_missions']):
        world=worlds[spec['world']]
        if spec.get('relocate'):
            old=world.shelves[world.objects[spec['target']]];world.relocate(spec['target'],next(r for r in world.rooms if r not in (world.home,old)))
        obs=world.reset(spec['target']);definition=world.audit_state();definition={k:definition[k] for k in ('world','home','edges','objects','shelves')};initial=copy.deepcopy(obs);events=[];trajectory=[];success=False;start=time.perf_counter()
        for _ in range(spec['budget']):
            memory.observe(obs)
            if actor:action,record=actor.act(obs,memory);trajectory.append(record)
            else:action=planner.act(obs,memory)
            previous=copy.deepcopy(obs);obs,reward=world.step(action);events.append(dict(observation=previous,action=action,next_observation=copy.deepcopy(obs),reward=reward))
            if reward:success=True;break
        if actor:torch.cuda.synchronize()
        episodes.append(dict(mission=mi,world_definition_hash=identity(definition),initial_observation=initial,success=success,actions=len(events),return_value=float(success)-.01*len(events),acting_seconds=time.perf_counter()-start,events=events,trajectory=trajectory,memory_state=memory.state(),memory_bytes=memory.bytes(),trajectory_utf8_bytes=len(canonical(trajectory).encode()),action_rng=copy.deepcopy(actor.action_rng.bit_generator.state) if actor else None))
    return dict(episodes=episodes,total_successes=sum(e['success'] for e in episodes),total_actions=sum(e['actions'] for e in episodes),acting_seconds=sum(e['acting_seconds'] for e in episodes),total_seconds=time.perf_counter()-start_all,work=work_delta(actor.work,before) if actor else {},final_memory_state=memory.state(),final_memory_bytes=memory.bytes())
