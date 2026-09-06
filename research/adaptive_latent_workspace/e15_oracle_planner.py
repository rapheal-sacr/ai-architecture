from pathlib import Path
import argparse,hashlib,inspect,json,math,subprocess,time
import numpy as np
import torch
from e14_world_model import Planner,normalized,inputs,targets,advance
HERE=Path(__file__).resolve().parent

class OracleDiagnostic:
    def __init__(self,gravity):self.gravity=gravity;self.forward_examples=0
    @torch.no_grad()
    def predict(self,x):
        self.forward_examples+=len(x)
        angle=torch.atan2(x[:,1],x[:,0]);velocity=8*x[:,2];torque=2*x[:,3]
        nxt=(velocity+(1.5*self.gravity*x[:,1]+3*torque)*.05).clamp(-8,8)
        delta=.05*nxt;cost=angle.square()+.1*velocity.square()+.001*torque.square()
        return torch.stack([torch.atan2(torch.sin(delta),torch.cos(delta))/math.pi,(nxt-velocity)/8,-cost/16],dim=1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e15.json').read_text());base=json.loads((HERE/'protocol_e14.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()==base['source_commit']
    import gymnasium as gym
    from gymnasium.envs.classic_control import pendulum
    assert hashlib.sha256(Path(inspect.getfile(pendulum)).read_bytes()).hexdigest()==base['pendulum_sha256']
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e14.json','e14_world_model.py','protocol_e15.json','e15_oracle_planner.py')}
    identity['gymnasium_commit']=base['source_commit'];a.out.mkdir(parents=True,exist_ok=True)
    checks=[]
    for gravity in dict.fromkeys(spec['gravity_sequence']):
        env=gym.make('Pendulum-v1',g=gravity);obs=normalized([env.reset(seed=77)[0]]);oracle=OracleDiagnostic(gravity);g=torch.Generator().manual_seed(78);maximum=0.
        for _ in range(100):
            action=2*torch.rand(1,generator=g)-1;pred=oracle.predict(inputs(obs,action));nxt,r,_,_,_=env.step(np.array([2*float(action[0])],dtype=np.float32));following=normalized([nxt]);truth=targets(obs,following,[r]);maximum=max(maximum,float((pred-truth).abs().max()));obs=following
        env.close();assert maximum<1e-5,(gravity,maximum);checks.append(dict(gravity=gravity,maximum_normalized_target_error=maximum,steps=100))
    (a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=checks),indent=2)+'\n');print(json.dumps(dict(preflight=checks)),flush=True)
    if not a.execute:return
    records=[]
    for seed in spec['seeds']:
        for stage,gravity in enumerate(spec['gravity_sequence']):
            for horizon in spec['horizons']:
                path=a.out/f'{seed}-stage{stage}-h{horizon}.json'
                if path.exists():r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
                eval_seed=seed+3000000+10000*stage
                envs=[gym.make('Pendulum-v1',g=gravity) for _ in range(spec['parallel_envs'])]
                obs=normalized([env.reset(seed=eval_seed+1000*i)[0] for i,env in enumerate(envs)])
                oracle=OracleDiagnostic(gravity);planner=Planner(oracle,base,len(envs),eval_seed+600000,horizon);returns=np.zeros(len(envs));start=time.perf_counter()
                for _ in range(spec['evaluation_steps']):
                    actions=planner.act(obs);next_obs=[]
                    for i,(env,action) in enumerate(zip(envs,actions.tolist())):
                        nxt,reward,_,_,_=env.step(np.array([2*action],dtype=np.float32));next_obs.append(nxt);returns[i]+=reward
                    obs=normalized(next_obs)
                for env in envs:env.close()
                r=dict(seed=seed,stage=stage,gravity=gravity,horizon=horizon,identity=identity,returns=returns.tolist(),mean_return=float(returns.mean()),
                    seconds=time.perf_counter()-start,actual_environment_steps=spec['evaluation_steps']*len(envs),planning_calls=planner.calls,oracle_model_examples=oracle.forward_examples)
                path.write_text(json.dumps(r,indent=2)+'\n');records.append(r);print(json.dumps({k:r[k] for k in ('seed','stage','gravity','horizon','mean_return','seconds')}),flush=True)
    (a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,preflight=checks,records=records),indent=2)+'\n');print('E15_COMPLETE',flush=True)
if __name__=='__main__':main()
