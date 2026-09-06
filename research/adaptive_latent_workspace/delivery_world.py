"""Persistent partially observed delivery world and observation-only controls.

World internals are never passed to policies. A pickup obtains a copy from a
stock shelf; stock remains until an exogenous relocation between missions.
"""
from collections import OrderedDict,deque
import copy,hashlib,json
import numpy as np

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def identity(x):return hashlib.sha256(canonical(x).encode()).hexdigest()

class DeliveryWorld:
    def __init__(self,seed,n,objects=2):
        self.seed=seed;self.n=n;rng=np.random.default_rng(seed);self.world='w'+''.join(rng.choice(list('abcdefghijkmnpqrstuvwxyz'),5))
        names=set()
        while len(names)<n:names.add('r'+''.join(rng.choice(list('abcdefghijkmnpqrstuvwxyz'),5)))
        self.rooms=sorted(names);order=list(rng.permutation(self.rooms));self.edges={r:set() for r in self.rooms}
        for i in range(n):
            a,b=order[i],order[(i+1)%n];self.edges[a].add(b);self.edges[b].add(a)
        # Optional sparse chords; degree remains at most three.
        for i in rng.permutation(n):
            candidates=[j for j in range(n) if j!=i and self.rooms[j] not in self.edges[self.rooms[i]] and len(self.edges[self.rooms[j]])<3]
            if len(self.edges[self.rooms[i]])<3 and candidates and rng.random()<.35:
                j=int(rng.choice(candidates));a,b=self.rooms[i],self.rooms[j];self.edges[a].add(b);self.edges[b].add(a)
        self.home=order[0];self.objects=['o'+''.join(rng.choice(list('abcdefghijkmnpqrstuvwxyz'),5)) for _ in range(objects)];assert len(set(self.objects))==objects
        choices=[r for r in self.rooms if r!=self.home];self.shelves={o:choices[int(rng.integers(len(choices)))] for o in self.objects};self.position=self.home;self.carrying=None;self.target=self.objects[0];self.done=False;self.steps=0;self.feedback='Mission has not started.'
    def reset(self,target_index):
        self.position=self.home;self.carrying=None;self.target=self.objects[target_index];self.done=False;self.steps=0;self.feedback='New mission. Stock remains on its usual shelf unless relocated.';return self.observe()
    def relocate(self,index,room):
        assert room in self.rooms and room!=self.home;self.shelves[self.objects[index]]=room
    def observe(self):
        items=sorted(o for o,r in self.shelves.items() if r==self.position)
        return dict(world=self.world,room=self.position,home=self.home,target=self.target,carrying=self.carrying,exits=sorted(self.edges[self.position]),items=items,feedback=self.feedback,done=self.done)
    def legal(self):return legal_actions(self.observe())
    def step(self,action):
        assert not self.done;assert action in self.legal(),f'Illegal action {action}'
        self.steps+=1;kind,arg=action.split(':',1)
        if kind=='move':self.position=arg;self.feedback='Moved to '+arg+'.'
        elif kind=='pick':self.carrying=arg;self.feedback='Picked up one copy of '+arg+'. Stock remains on the shelf.'
        elif kind=='deliver':assert self.position==self.home and self.carrying==self.target;self.done=True;self.feedback='Delivery succeeded.';self.carrying=None
        elif kind=='drop':self.feedback='Discarded the carried copy.';self.carrying=None
        elif kind=='wait':self.feedback='Waited.'
        else:raise AssertionError(kind)
        return self.observe(),float(self.done)
    def audit_state(self):
        return dict(seed=self.seed,world=self.world,home=self.home,edges={k:sorted(v) for k,v in self.edges.items()},objects=self.objects,shelves=self.shelves,position=self.position,carrying=self.carrying,target=self.target,done=self.done,steps=self.steps,feedback=self.feedback)

def legal_actions(obs):
    actions=['move:'+r for r in obs['exits']]
    if obs['carrying'] is None:actions+=['pick:'+o for o in obs['items']]
    else:
        if obs['room']==obs['home'] and obs['carrying']==obs['target']:actions+=['deliver:'+obs['target']]
        actions+=['drop:'+obs['carrying']]
    actions+=['wait:here'];assert len(actions)<=8;return actions

class ObservationMemory:
    def __init__(self,capacity=None):self.capacity=capacity;self.records=OrderedDict();self.writes=0;self.evictions=0
    def observe(self,obs):
        key=(obs['world'],obs['room']);self.records.pop(key,None);self.records[key]=dict(exits=list(obs['exits']),items=list(obs['items']));self.writes+=1
        if self.capacity is not None:
            while len(self.records)>self.capacity:self.records.popitem(last=False);self.evictions+=1
    def view(self,world):return {room:copy.deepcopy(data) for (w,room),data in self.records.items() if w==world}
    def state(self):return dict(capacity=self.capacity,writes=self.writes,evictions=self.evictions,records=[dict(world=w,room=r,**d) for (w,r),d in self.records.items()])
    def bytes(self):return len(canonical(self.state()).encode())

class ObservedPlanner:
    """Exact BFS over observed edges; systematic exploration, no hidden map."""
    def act(self,obs,memory):
        actions=legal_actions(obs);room=obs['room'];target=obs['target']
        if obs['carrying']==target:
            if room==obs['home']:return 'deliver:'+target
            desired={obs['home']}
        elif obs['carrying'] is not None:return 'drop:'+obs['carrying']
        elif target in obs['items']:return 'pick:'+target
        else:
            known=memory.view(obs['world']);desired={r for r,d in known.items() if target in d['items']}
        known=memory.view(obs['world']);edges={}
        for r,d in known.items():
            for other in d['exits']:edges.setdefault(r,set()).add(other);edges.setdefault(other,set()).add(r)
        if not desired:
            desired=set(edges)-set(known)
            if not desired:desired=set(list(r for r in known if r!=room)[:1])
        queue=deque([(room,None)]);seen={room}
        while queue:
            here,first=queue.popleft()
            if here in desired and first is not None:return 'move:'+first
            for other in sorted(edges.get(here,())):
                if other not in seen:seen.add(other);queue.append((other,other if first is None else first))
        return next((a for a in actions if a.startswith('move:')),'wait:here')

RULES=('You control a delivery robot. Corridors are bidirectional. Each mission starts at home. '
       'Find the requested object, pick up a copy, return home, and deliver it. Picking up does not remove stock. '
       'Some shelves can change between missions. Only observations are reliable. '
       'Use remembered exits to navigate. Explore rooms whose contents you do not know when the target is unknown. '
       'Do not discard the requested object. Choose exactly one listed option letter.')

def render_prompt(obs,memory):
    known=memory.view(obs['world']);lines=['Remembered observations in this world, oldest to newest:']
    for room,d in known.items():lines.append(room+' | exits '+','.join(d['exits'])+' | stock '+(','.join(d['items']) or 'none'))
    lines+=['Current world: '+obs['world'],'Current room: '+obs['room'],'Home: '+obs['home'],'Requested object: '+obs['target'],'Carrying: '+str(obs['carrying']),'Visible stock: '+(','.join(obs['items']) or 'none'),'Feedback: '+obs['feedback'],'Options:']
    actions=legal_actions(obs)
    for i,action in enumerate(actions):lines.append(chr(65+i)+'. '+action)
    lines.append('Answer with one option letter only.');return '\n'.join(lines),actions
