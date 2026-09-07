"""Finite-planner/environment mechanics; no trained action competence claim."""
import argparse
import io
import itertools
import json
import time
from pathlib import Path

import torch

from action_e2e import ActionE2E
from action_navigation import graph, NavigationWorld, NavigationObservation, ObservedTransitionPlanner
from action_planning import PlanningConfig, goal_values, transition_logits, choose_distribution
from e2e_core import Config, E2ECore
from preflight_action_e2e import assert_action_state_equal


def run(out):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result = {}
    # Exhaustively enumerate all action strings on deterministic graphs and
    # terminate at first arrival. This bypasses the Bellman implementation.
    comparisons, max_error = 0, 0.
    for seed in range(36301,36307):
        mapping = graph(seed,4,2)
        probability = torch.nn.functional.one_hot(torch.tensor(mapping),4).double()
        for horizon in [1,2,4]:
            cfg = PlanningConfig(horizon=horizon,discount=.8)
            for goal in range(4):
                values = goal_values(probability,goal,cfg)
                for state in range(4):
                    for action in range(2):
                        scores=[]
                        for remaining in itertools.product(range(2),repeat=horizon-1):
                            here, score = state, 0.
                            for t,a in enumerate((action,)+remaining):
                                here = mapping[here][a]
                                if here==goal:
                                    score = .8**t
                                    break
                            scores.append(score)
                        diff=abs(values[state,action].item()-max(scores))
                        max_error=max(max_error,diff)
                        assert diff<1e-12
                        comparisons+=1
    result['exhaustive_deterministic_values']=comparisons
    result['exhaustive_value_max_error']=max_error

    # A stochastic reference maximizes conditional continuations separately
    # after each observed successor. It uses Python floats and explicit branches.
    torch.manual_seed(36311)
    probs=torch.rand(4,2,4,dtype=torch.float64)
    probs/=probs.sum(-1,keepdim=True)
    p=probs.tolist()
    def reference(state,action,goal,depth):
        total=0.
        for y,chance in enumerate(p[state][action]):
            future=max(reference(y,a,goal,depth-1) for a in range(2)) if depth>1 and y!=goal else 0.
            total+=chance*(1. if y==goal else .8*future)
        return total
    errors=[]
    for goal in range(4):
        q=goal_values(probs,goal,PlanningConfig(horizon=3,discount=.8))
        errors.extend(abs(q[s,a].item()-reference(s,a,goal,3)) for s in range(4) for a in range(2))
    assert max(errors)<1e-12
    result['stochastic_branch_reference_max_error']=max(errors)

    # Full observation-only table should choose a shortest-path first action.
    mapping=graph(36321,6,2)
    table=ObservedTransitionPlanner(6,2)
    for s,row in enumerate(mapping):
        for a,y in enumerate(row):
            table.observe_transition(s,a,y)
    correct=0
    for s in range(6):
        for goal in range(6):
            if s==goal:
                continue
            q=goal_values(torch.nn.functional.one_hot(torch.tensor(mapping),6).double(),goal,
                          PlanningConfig(horizon=6,discount=.8))
            a=table.act(NavigationObservation(s,goal))
            assert q[s,a] == q[s].max()
            correct+=1
    result['observed_map_shortest_path_actions']=correct

    # Serialize both world and partial table during hidden context recurrence.
    maps=[mapping,graph(36322,6,2)]
    world=NavigationWorld(maps,[13,17,19],[0,1,0],goal_seed=36323)
    learner=ObservedTransitionPlanner(6,2,capacity=7)
    for _ in range(11):
        old=world.observe()
        action=learner.act(old)
        obs,reward,receipt=world.step(action)
        learner.observe_transition(old.state,action,obs.state)
    buf=io.BytesIO()
    torch.save(dict(world=world.payload(),learner=learner.payload()),buf)
    buf.seek(0)
    saved=torch.load(buf,weights_only=False)
    restored=NavigationWorld.restore(saved['world'])
    learner2=ObservedTransitionPlanner.restore(saved['learner'])
    for t in range(11,49):
        before=world.observe()
        assert before==restored.observe()
        a=learner.act(before)
        assert a==learner2.act(before)
        first=world.step(a)
        second=restored.step(a)
        assert first==second
        obs,reward,receipt=first
        assert receipt['reward']==int(obs.state==before.goal)
        slot=0 if t<13 or t>=30 else 1
        assert obs.state==maps[slot][before.state][a]
        assert set(receipt)=={'step','observation','goal','executed_action','outcome','reward','next_goal'}
        learner.observe_transition(before.state,a,obs.state)
        learner2.observe_transition(before.state,a,obs.state)
        assert learner.payload()==learner2.payload()
    result['serialized_world_and_table_exact_steps']=38

    cfg=Config(vocab=8,width=16,hidden=24,layers=3,suffix=1,heads=2,window=4,
               chunk=2,inner_lr=.3,clip=1.,init_std=.15)
    core=E2ECore(cfg,seed=36331,dtype=torch.float64)
    adapter=ActionE2E(core,states=6,actions=2)
    state=adapter.initial_state(0)
    for action,outcome in [(0,1),(1,3),(0,2)]:
        _,state,_,_=adapter.observe(adapter.forecast(state,action),outcome)
    saved=adapter.restore(adapter.payload(state.detached()))
    vector,work=transition_logits(adapter,state)
    serial,serial_work=transition_logits(adapter,state,vectorized=False)
    assert torch.allclose(vector,serial,atol=2e-12,rtol=2e-12)
    assert_action_state_equal(state.detached(),saved)
    assert work['hypothetical_queries']==12 and work['forward_tokens']==24 and work['inner_updates']==0
    result['vector_serial_max_error']=float((vector-serial).abs().max().detach())
    result['hypothetical_work']=work
    pcfg=PlanningConfig(horizon=3,temperature=.3)
    vpolicy,_=choose_distribution(adapter,state,4,pcfg)
    spolicy,_=choose_distribution(adapter,state,4,pcfg,vectorized=False)
    gv=torch.autograd.grad(vpolicy[0],tuple(core.p.values()),retain_graph=True)
    gs=torch.autograd.grad(spolicy[0],tuple(core.p.values()),retain_graph=True)
    err=max((a-b).abs().max().item() for a,b in zip(gv,gs))
    assert err<2e-10
    result['vector_serial_outer_gradient_max_error']=err

    # Entire outer objective includes a real fast update followed by model
    # queries and planning. Perturb all initial parameters along one direction.
    names=list(core.p)
    torch.manual_seed(36332)
    direction={k:torch.randn_like(v) for k,v in core.p.items()}
    norm=sum(d.square().sum() for d in direction.values()).sqrt()
    direction={k:d/norm for k,d in direction.items()}
    def objective(params):
        initial=adapter.initial_state(0,params=params)
        forecast=adapter.forecast(initial,0,params=params)
        _,updated,_,_=adapter.observe(forecast,2,mode='second_order')
        policy,_=choose_distribution(adapter,updated,4,pcfg,params=params)
        return policy[0].log()
    loss=objective(core.p)
    gradients=torch.autograd.grad(loss,tuple(core.p.values()))
    autodiff=sum((g*direction[k]).sum() for k,g in zip(names,gradients)).item()
    deviations=[]
    for epsilon in [1e-4,1e-5]:
        plus={k:(v.detach()+epsilon*direction[k]).requires_grad_() for k,v in core.p.items()}
        minus={k:(v.detach()-epsilon*direction[k]).requires_grad_() for k,v in core.p.items()}
        difference=(objective(plus).item()-objective(minus).item())/(2*epsilon)
        deviations.append(abs(difference-autodiff))
    assert max(deviations)<1e-7
    result['full_outer_directional_derivative']=autodiff
    result['finite_difference_absolute_errors']=deviations
    result['real_state_unchanged_after_planning']=True
    result['status']='ACTION_PLANNING_PREFLIGHT_PASS'
    result['seconds']=time.perf_counter()-started
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args().out)
