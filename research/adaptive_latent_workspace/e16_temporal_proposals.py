from pathlib import Path
import argparse,hashlib,inspect,json,time
import numpy as np
import torch
from e14_world_model import Planner,normalized,advance
from e15_oracle_planner import OracleDiagnostic
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class BlockPlanner(Planner):
    def __init__(self,*args,block,**kwargs):
        super().__init__(*args,**kwargs);self.block=block;assert self.h%block==0
    def expand(self,coefficients):return coefficients.repeat_interleave(self.block,dim=-1)
    @torch.no_grad()
    def act(self,obs):
        if self.block==1:return super().act(obs)
        c=self.cfg;n,p,h=self.n,c['planner_population'],self.h;k=h//self.block
        mean=self.mean.reshape(n,k,self.block).mean(-1);std=torch.ones_like(mean);before=self.learner.forward_examples
        for _ in range(c['planner_iterations']):
            coefficients=(mean[:,None,:]+std[:,None,:]*torch.randn(n,p,k,generator=self.rng)).clamp(-1,1)
            coefficients[:,0]=mean;coefficients[:,1]=0;actions=self.expand(coefficients)
            state=obs[:,None,:].expand(n,p,3).reshape(n*p,3).clone();scores=torch.zeros(n,p)
            for t in range(h):
                state,reward=advance(self.learner,state,actions[:,:,t].reshape(-1));scores+=c['planner_discount']**t*reward.reshape(n,p)
            ids=scores.topk(c['planner_elites'],dim=1).indices;elites=coefficients.gather(1,ids[:,:,None].expand(-1,-1,k))
            mean=elites.mean(1);std=elites.std(1,unbiased=False).clamp_min(.1);best=actions[torch.arange(n),scores.argmax(1)].clone()
        expanded=self.expand(mean);self.mean=torch.cat([expanded[:,1:],torch.zeros(n,1)],dim=1)
        self.calls+=1;self.model_examples+=self.learner.forward_examples-before;return best[:,0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e16.json').read_text());base=json.loads((HERE/'protocol_e14.json').read_text())
    import gymnasium as gym
    from gymnasium.envs.classic_control import pendulum
    assert gym.__version__=='1.1.1' and digest(inspect.getfile(pendulum))==base['pendulum_sha256']
    prior=json.loads((a.runs/'e15/complete.json').read_text());original=json.loads((a.runs/'e14/complete.json').read_text())
    for n,h in original['identity'].items():
        if n.endswith(('.py','.json')):assert digest(HERE/n)==h
    for n,h in prior['identity'].items():
        if n.endswith(('.py','.json')):assert digest(HERE/n)==h
    identity={n:digest(HERE/n) for n in ('protocol_e14.json','e14_world_model.py','protocol_e15.json','e15_oracle_planner.py','protocol_e16.json','e16_temporal_proposals.py')}
    identity['e14_identity']=original['identity'];identity['e15_identity']=prior['identity'];a.out.mkdir(parents=True,exist_ok=True)
    checks=[];dummy=torch.tensor([[1.,0.,.1],[0.,1.,-.2],[-1.,0.,0.],[0.,-1.,.3]])
    cfg=base.copy();cfg.update(planner_population=8,planner_elites=2,planner_iterations=1)
    old=Planner(OracleDiagnostic(10),cfg,4,77,60);new=BlockPlanner(OracleDiagnostic(10),cfg,4,77,60,block=1)
    for _ in range(3):assert torch.equal(old.act(dummy),new.act(dummy)) and torch.equal(old.mean,new.mean)
    for block in spec['blocks']:
        predictor=OracleDiagnostic(10);planner=BlockPlanner(predictor,cfg,4,78,60,block=block)
        coefficients=torch.arange(60//block,dtype=torch.float32)[None,:];expanded=planner.expand(coefficients)
        assert torch.equal(expanded.reshape(1,60//block,block),coefficients[:,:,None].expand(-1,-1,block))
        action=planner.act(dummy);assert action.abs().max()<=1 and torch.isfinite(action).all()
        assert predictor.forward_examples==4*8*60
        checks.append(dict(block=block,forward_examples=predictor.forward_examples,within_block_constant=True))
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,block_one_exact=True,checks=checks),indent=2)+'\n');print('E16 proposal preflight passed',flush=True)
    if not a.execute:return
    records=[]
    for seed in spec['seeds']:
        for stage,gravity in enumerate(spec['gravity_sequence']):
            for mode in spec['modes']:
                for block in spec['blocks']:
                    path=a.out/f'{seed}-stage{stage}-{mode}-block{block}.json'
                    if path.exists():r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
                    if mode=='oracle_diagnostic' and block==1:
                        source=next(x for x in prior['records'] if x['seed']==seed and x['stage']==stage and x['horizon']==60)
                        r=dict(seed=seed,stage=stage,gravity=gravity,mode=mode,block=block,identity=identity,returns=source['returns'],mean_return=source['mean_return'],new_environment_steps=0,new_model_examples=0,
                            inherited_environment_steps=source['actual_environment_steps'],inherited_model_examples=source['oracle_model_examples'],seconds=source['seconds'],reused_e15_record=source)
                    else:
                        started=time.perf_counter();checkpoint=None;learner_costs=None
                        if mode=='oracle_diagnostic':predictor=OracleDiagnostic(gravity)
                        else:
                            checkpoint=a.runs/'e14'/f'{seed}-replay_mpc20-stage{stage}.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
                            assert payload['identity']==original['identity'];predictor=payload['learner'];learner_costs=predictor.costs()
                        restored=time.perf_counter()-started;eval_seed=seed+3000000+10000*stage
                        envs=[gym.make('Pendulum-v1',g=gravity) for _ in range(4)];obs=normalized([env.reset(seed=eval_seed+1000*i)[0] for i,env in enumerate(envs)])
                        planner=BlockPlanner(predictor,base,4,eval_seed+600000,spec['horizon'],block=block);returns=np.zeros(4)
                        for _ in range(spec['evaluation_steps']):
                            actions=planner.act(obs);following=[]
                            for i,(env,action) in enumerate(zip(envs,actions.tolist())):
                                nxt,reward,_,_,_=env.step(np.array([2*action],dtype=np.float32));following.append(nxt);returns[i]+=reward
                            obs=normalized(following)
                        for env in envs:env.close()
                        assert planner.model_examples==4*base['planner_population']*base['planner_iterations']*spec['horizon']*spec['evaluation_steps']
                        r=dict(seed=seed,stage=stage,gravity=gravity,mode=mode,block=block,identity=identity,returns=returns.tolist(),mean_return=float(returns.mean()),new_environment_steps=4*spec['evaluation_steps'],
                            new_model_examples=planner.model_examples,inherited_environment_steps=0,inherited_model_examples=0,seconds=time.perf_counter()-started,restore_seconds=restored,
                            checkpoint=str(checkpoint) if checkpoint else None,checkpoint_sha256=digest(checkpoint) if checkpoint else None,learner_costs=learner_costs)
                    path.write_text(json.dumps(r,indent=2)+'\n');records.append(r)
                    print(json.dumps({k:r[k] for k in ('seed','stage','mode','block','mean_return')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n');print('E16_COMPLETE',flush=True)
if __name__=='__main__':main()
