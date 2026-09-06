"""Observation-only E31 reference sets and label-preserving prompt transforms."""
from collections import deque,OrderedDict
import copy,re
import numpy as np
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,render_prompt,legal_actions,identity,canonical

def reference(obs,memory):
    actions=legal_actions(obs);room=obs['room'];target=obs['target'];known=memory.view(obs['world'])
    if obs['carrying']==target and room==obs['home']:return dict(category='deliver',indices=[actions.index('deliver:'+target)],distance=0)
    if obs['carrying'] is not None and obs['carrying']!=target:return dict(category='drop_wrong',indices=[actions.index('drop:'+obs['carrying'])],distance=0)
    if obs['carrying'] is None and target in obs['items']:return dict(category='pick_target',indices=[actions.index('pick:'+target)],distance=0)
    edges={}
    for r,data in known.items():
        for other in data['exits']:edges.setdefault(r,set()).add(other);edges.setdefault(other,set()).add(r)
    if obs['carrying']==target:goals={obs['home']};kind='home'
    else:
        goals={r for r,data in known.items() if target in data['items']};kind='stock'
        if not goals:goals=set(edges)-set(known);kind='explore'
        if not goals:goals=set(list(r for r in known if r!=room)[:1]);kind='reinspect'
    distances={g:0 for g in goals};queue=deque(sorted(goals))
    while queue:
        here=queue.popleft()
        for other in sorted(edges.get(here,())):
            if other not in distances:distances[other]=distances[here]+1;queue.append(other)
    d=distances.get(room)
    if d is not None and d>0:
        good=['move:'+r for r in obs['exits'] if distances.get(r)==d-1];assert good
        category=kind+('_one_edge' if d==1 else '_multi_edge');return dict(category=category,indices=[actions.index(a) for a in good],distance=d)
    fallback=next((a for a in actions if a.startswith('move:')),'wait:here')
    return dict(category='fallback',indices=[actions.index(fallback)],distance=None)

def memory_from_state(state):
    m=ObservationMemory(state['capacity']);m.writes=state['writes'];m.evictions=state['evictions']
    for r in state['records']:m.records[(r['world'],r['room'])]=dict(exits=list(r['exits']),items=list(r['items']))
    return m

def query(obs,memory,action,split,index):
    prompt,actions=render_prompt(obs,memory);ref=reference(obs,memory);assert actions.index(action) in ref['indices']
    return dict(split=split,index=index,observation=copy.deepcopy(obs),memory=memory.state(),prompt=prompt,actions=actions,teacher_index=actions.index(action),reference=ref)

def generate(seed,source_bootstrap,source_cfg,cfg,preflight=False):
    planner=ObservedPlanner();full=ObservationMemory();bounded=ObservationMemory(source_cfg['bounded_room_records']);tutorial=[];original=[]
    for i in range(source_cfg['bootstrap_worlds']):
        world=DeliveryWorld(seed+700000+i,source_cfg['bootstrap_world_size'])
        for mi in range(source_cfg['bootstrap_missions_per_world']):
            obs=world.reset(mi%2);success=False
            for _ in range(source_cfg['bootstrap_steps_per_mission']):
                full.observe(obs);bounded.observe(obs);assert full.view(obs['world'])==bounded.view(obs['world']);action=planner.act(obs,full);q=query(obs,bounded,action,'tutorial',len(tutorial));tutorial.append(q);original.append(dict(prompt=q['prompt'],actions=q['actions'],index=q['teacher_index'],observation=copy.deepcopy(obs)));obs,reward=world.step(action)
                if reward:success=True;break
            assert success
    assert original==source_bootstrap['examples'] and identity(original)==source_bootstrap['example_sha256']
    exposure=np.zeros(len(tutorial),dtype=np.int64);rng=np.random.default_rng(seed+800000)
    for _ in range(source_cfg['bootstrap_updates']):exposure[int(rng.integers(len(tutorial)))]+=1
    for q,n in zip(tutorial,exposure):q['bootstrap_index_exposures']=int(n)
    # Identical prompt/label records at different indices share training exposure.
    pooled={}
    for q in tutorial:
        key=identity([q['prompt'],q['teacher_index']]);pooled[key]=pooled.get(key,0)+q['bootstrap_index_exposures']
    for q in tutorial:q['bootstrap_prompt_label_exposures']=pooled[identity([q['prompt'],q['teacher_index']])]
    fresh=[];mem=ObservationMemory();teacher_actions=0
    for wi,n in enumerate(cfg['fresh_world_sizes']):
        world=DeliveryWorld(seed+cfg['fresh_world_seed_offset']+wi,n)
        for mi in range(cfg['fresh_missions_per_world']):
            obs=world.reset(mi%2);success=False
            for _ in range(cfg['fresh_steps_per_room']*n):
                mem.observe(obs);action=planner.act(obs,mem);q=query(obs,mem,action,'fresh',len(fresh));q.update(world_size=n,bootstrap_index_exposures=0,bootstrap_prompt_label_exposures=0);fresh.append(q);teacher_actions+=1;obs,reward=world.step(action)
                if reward:success=True;break
            assert success
    if preflight:tutorial=tutorial[:8];fresh=fresh[:8]
    return tutorial+fresh,dict(tutorial_source_queries=len(original),fresh_teacher_actions=teacher_actions,tutorial_scored_queries=len(tutorial),fresh_scored_queries=len(fresh))

def rename(q,encoding,seed):
    if encoding=='original':return dict(prompt=q['prompt'],actions=q['actions'],mapping={},reference=q['reference'])
    obs=q['observation'];mem=memory_from_state(q['memory']);labels={obs['world'],obs['room'],obs['home'],obs['target'],*obs['exits'],*obs['items']}
    if obs['carrying']:labels.add(obs['carrying'])
    # Only the current-world view enters a prompt, so other stored worlds cannot influence the bijection.
    for room,data in mem.view(obs['world']).items():labels.add(room);labels.update(data['exits']);labels.update(data['items'])
    groups={p:sorted(v for v in labels if v.startswith(p)) for p in ('w','r','o')};rng=np.random.default_rng(seed);mapping={};used=set(labels)
    for prefix,names in groups.items():
        for i,name in enumerate(names):
            if encoding=='compact':replacement=prefix+str(i)
            else:
                assert encoding=='fresh_nonce'
                while True:
                    replacement=prefix+''.join(rng.choice(list('abcdefghijkmnpqrstuvwxyz'),5))
                    if replacement not in used:break
            used.add(replacement);mapping[name]=replacement
    assert len(set(mapping.values()))==len(mapping)
    pattern=re.compile(r'\b(?:'+'|'.join(re.escape(k) for k in sorted(mapping))+r')\b');transform=lambda s:pattern.sub(lambda m:mapping[m.group()],s)
    prompt=transform(q['prompt']);actions=[transform(a) for a in q['actions']];inverse={v:k for k,v in mapping.items()};invpattern=re.compile(r'\b(?:'+'|'.join(re.escape(k) for k in sorted(inverse))+r')\b');assert invpattern.sub(lambda m:inverse[m.group()],prompt)==q['prompt']
    newobs=copy.deepcopy(obs)
    for key in ('world','room','home','target','carrying'):
        if newobs[key] is not None:newobs[key]=mapping[newobs[key]]
    for key in ('exits','items'):newobs[key]=[mapping[x] for x in newobs[key]]
    newobs['feedback']=transform(newobs['feedback']);newmem=ObservationMemory(mem.capacity);newmem.writes=mem.writes;newmem.evictions=mem.evictions
    for (world,room),data in mem.records.items():
        if world==obs['world']:newmem.records[(mapping[world],mapping[room])]=dict(exits=[mapping[x] for x in data['exits']],items=[mapping[x] for x in data['items']])
    assert render_prompt(newobs,newmem)==(prompt,actions);ref=reference(newobs,newmem);assert ref==q['reference']
    return dict(prompt=prompt,actions=actions,mapping=mapping,reference=ref)
