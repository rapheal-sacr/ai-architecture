from pathlib import Path
import argparse,copy,json,hashlib
from delivery_world import DeliveryWorld,ObservationMemory,ObservedPlanner,render_prompt,identity

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();checks=[];policy=ObservedPlanner()
    for seed in (1611,1612,1613):
        for n in (4,8,16):
            world=DeliveryWorld(seed,n);mem=ObservationMemory();counts=[]
            for mission in range(6):
                if mission==4:
                    old=world.shelves[world.objects[0]];world.relocate(0,next(r for r in world.rooms if r not in (world.home,old)))
                obs=world.reset(mission%2);success=False
                for step in range(8*n):
                    mem.observe(obs);before=identity(world.audit_state());action=policy.act(obs,mem);render_prompt(obs,mem);assert identity(world.audit_state())==before;obs,reward=world.step(action)
                    if reward:success=True;break
                assert success;(counts.append(step+1))
            checks.append(dict(seed=seed,rooms=n,missions=6,all_success=True,action_counts=counts))
    # Hidden worlds differ beyond the current view; identical observable history
    # must produce identical memory, prompt and observed-only planner decision.
    w=DeliveryWorld(1614,8);obs=w.reset(0);other=copy.deepcopy(w);hidden=[r for r in w.rooms if r!=w.home];other.relocate(0,next(r for r in hidden if r!=w.shelves[w.objects[0]]));assert w.observe()==other.observe();m1=ObservationMemory(4);m2=ObservationMemory(4);m1.observe(w.observe());m2.observe(other.observe());assert m1.state()==m2.state();assert render_prompt(w.observe(),m1)==render_prompt(other.observe(),m2);assert policy.act(w.observe(),m1)==policy.act(other.observe(),m2)
    # Capacity evicts records across worlds, never merely hides a full archive.
    memory=ObservationMemory(2)
    for i in range(5):
        q=copy.deepcopy(obs);q['room']='room'+str(i);memory.observe(q)
    assert len(memory.records)==2 and memory.evictions==3 and len(memory.state()['records'])==2
    # Environment action outcomes really change subsequent local observations.
    obs=w.reset(0);move='move:'+obs['exits'][0];following,reward=w.step(move);assert following['room']==move.split(':')[1] and following!=obs and reward==0
    result=dict(status='ENVIRONMENT_PREFLIGHT_ONLY_NO_NEURAL_AGENT_RESULT',planner_cases=checks,hidden_state_noninterference=True,capacity_really_evicts=True,action_changes_observation=True,source_hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in ['delivery_world.py','preflight_delivery_world.py']})
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
