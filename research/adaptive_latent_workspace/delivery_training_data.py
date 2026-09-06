"""Domain-separated tutorial and evaluation worlds, with no shared seed ranges."""
import hashlib,copy
import numpy as np
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,identity
from delivery_diagnostics import query

def seed_for(seed,domain,index=0):
    return int.from_bytes(hashlib.sha256(f'E32|{seed}|{domain}|{index}'.encode()).digest()[:8],'big')

def make_data(seed,cfg):
    planner=ObservedPlanner();memory=ObservationMemory(cfg['tutorial_rooms']);examples=[];queries=[];missions=[];worlds=[]
    for wi in range(cfg['tutorial_worlds']):
        world_seed=seed_for(seed,'tutorial',wi);world=DeliveryWorld(world_seed,cfg['tutorial_rooms']);worlds.append(dict(seed=world_seed,world=world.world,rooms=world.n,domain='tutorial',index=wi))
        for mi in range(cfg['tutorial_missions_per_world']):
            obs=world.reset(mi%2);success=False;steps=0
            for _ in range(cfg['tutorial_steps_per_room']*world.n):
                memory.observe(obs);action=planner.act(obs,memory);q=query(obs,memory,action,'tutorial',len(queries));queries.append(q);examples.append(dict(prompt=q['prompt'],actions=q['actions'],index=q['teacher_index'],observation=copy.deepcopy(obs)));obs,reward=world.step(action);steps+=1
                if reward:success=True;break
            assert success;missions.append(dict(world=world.world,mission=mi,actions=steps))
    for wi,n in enumerate(cfg['eval_world_sizes']):
        world_seed=seed_for(seed,'evaluation',wi);world=DeliveryWorld(world_seed,n);worlds.append(dict(seed=world_seed,world=world.world,rooms=n,domain='evaluation',index=wi))
    rng=np.random.default_rng(seed_for(seed,'training_order'));order=[int(x) for x in rng.integers(len(examples),size=max(cfg['milestones']))]
    return dict(seed=seed,worlds=worlds,examples=examples,queries=queries,teacher_missions=missions,teacher_actions=sum(m['actions'] for m in missions),order=order,example_sha256=identity(examples),order_sha256=identity(order))

def assert_disjoint(datasets):
    specs=[(w['seed'],w['rooms']) for d in datasets for w in d['worlds']];seeds=[w['seed'] for d in datasets for w in d['worlds']];labels=[w['world'] for d in datasets for w in d['worlds']];assert len(set(specs))==len(specs) and len(set(seeds))==len(seeds) and len(set(labels))==len(labels)
    return dict(worlds=len(specs),all_seed_specs_disjoint=True,all_world_labels_distinct=True)
