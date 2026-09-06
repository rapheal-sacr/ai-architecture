from pathlib import Path
import argparse,copy,hashlib,inspect,json,math,subprocess,time
import numpy as np
import torch
import e1_stream as e1
from e6_sharing import GuardedSharing
HERE=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def normalized(obs):
    out=torch.as_tensor(np.array(obs),dtype=torch.float32).clone();out[:,2]/=8.;return out

def inputs(obs,actions):return torch.cat([obs,actions[:,None]],dim=1)
def targets(obs,next_obs,rewards):
    a=torch.atan2(obs[:,1],obs[:,0]);b=torch.atan2(next_obs[:,1],next_obs[:,0]);delta=b-a
    return torch.stack([torch.atan2(torch.sin(delta),torch.cos(delta))/math.pi,next_obs[:,2]-obs[:,2],torch.as_tensor(rewards,dtype=torch.float32)/16.],dim=1)

@torch.no_grad()
def advance(learner,state,actions):
    pred=learner.predict(inputs(state,actions));assert torch.isfinite(pred).all(),'Nonfinite learned transition'
    angle=torch.atan2(state[:,1],state[:,0])+math.pi*pred[:,0]
    return torch.stack([torch.cos(angle),torch.sin(angle),(state[:,2]+pred[:,1]).clamp(-1,1)],dim=1),16.*pred[:,2]

class Planner:
    def __init__(self,learner,cfg,n,seed,horizon):
        self.learner=learner;self.cfg=cfg;self.n=n;self.h=horizon;self.mean=torch.zeros(n,horizon)
        self.rng=torch.Generator().manual_seed(seed);self.calls=0;self.model_examples=0
    def reset(self,indices):self.mean[indices]=0
    @torch.no_grad()
    def act(self,obs):
        c=self.cfg;n,p,h=self.n,c['planner_population'],self.h;mean=self.mean.clone();std=torch.ones_like(mean)
        before=self.learner.forward_examples
        for _ in range(c['planner_iterations']):
            noise=torch.randn(n,p,h,generator=self.rng)
            actions=(mean[:,None,:]+std[:,None,:]*noise).clamp(-1,1);actions[:,0]=mean;actions[:,1]=0
            state=obs[:,None,:].expand(n,p,3).reshape(n*p,3).clone();scores=torch.zeros(n,p)
            for t in range(h):
                state,reward=advance(self.learner,state,actions[:,:,t].reshape(-1));scores+=c['planner_discount']**t*reward.reshape(n,p)
            elite_ids=scores.topk(c['planner_elites'],dim=1).indices
            elite=actions.gather(1,elite_ids[:,:,None].expand(-1,-1,h));mean=elite.mean(1);std=elite.std(1,unbiased=False).clamp_min(.1)
            best=actions[torch.arange(n),scores.argmax(1)].clone()
        self.mean=torch.cat([mean[:,1:],torch.zeros(n,1)],dim=1);self.calls+=1
        self.model_examples+=self.learner.forward_examples-before
        return best[:,0]

def create_learner(arm,cfg):
    if arm=='random':return None
    if arm=='guarded_mpc20':return GuardedSharing(cfg,torch.device('cpu'))
    return e1.SingleLearner(cfg,torch.device('cpu'),'context_replay' if arm=='context_mpc20' else 'replay')

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e14.json').read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repo,text=True).strip()==spec['source_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.repo,text=True).strip()
    import gymnasium as gym
    from gymnasium.envs.classic_control import pendulum
    assert gym.__version__=='1.1.1'
    assert digest(inspect.getfile(pendulum))==spec['pendulum_sha256']==digest(a.repo/'gymnasium/envs/classic_control/pendulum.py')
    cfg=json.loads((HERE/'protocol_e1.json').read_text());cfg.update(json.loads((HERE/'protocol_e2.json').read_text()));cfg.update(json.loads((HERE/'protocol_e6.json').read_text()));cfg.update(input_dim=4,output_dim=3)
    identity={n:digest(HERE/n) for n in ('protocol_e1.json','protocol_e2.json','protocol_e6.json','protocol_e14.json','e1_stream.py','e2_isolation.py','e5_controls.py','e6_sharing.py','e14_world_model.py')}
    identity.update(gymnasium_commit=spec['source_commit'],gymnasium_version=gym.__version__,pendulum_sha256=spec['pendulum_sha256'],torch_version=torch.__version__)
    if not a.execute:
        spec=copy.deepcopy(spec);spec.update(seeds=[77],gravity_sequence=[10.],steps_per_stage=16,initial_random_steps=4,
            evaluation_steps=8,probe_steps=4,planner_horizon=3,planner_population=8,planner_elites=2,planner_iterations=1)
    a.out.mkdir(parents=True,exist_ok=True)
    def envs_for(gravity,seed):
        envs=[gym.make('Pendulum-v1',g=gravity) for _ in range(spec['parallel_envs'])]
        obs=[env.reset(seed=seed+1000*i)[0] for i,env in enumerate(envs)];return envs,normalized(obs)
    def step_envs(envs,actions):
        next_obs=[];rewards=[];done=[]
        for env,action in zip(envs,actions.tolist()):
            obs,r,term,trunc,_=env.step(np.array([2*action],dtype=np.float32));next_obs.append(obs);rewards.append(r);done.append(term or trunc)
        return normalized(next_obs),rewards,done
    def evaluate(learner,arm,gravity,seed):
        envs,obs=envs_for(gravity,seed);n=len(envs);total=np.zeros(n);g=torch.Generator().manual_seed(seed+500000)
        planner=None if learner is None else Planner(learner,spec,n,seed+600000,1 if arm=='replay_mpc1' else spec['planner_horizon'])
        start=time.perf_counter()
        for _ in range(spec['evaluation_steps']):
            actions=2*torch.rand(n,generator=g)-1 if planner is None else planner.act(obs)
            obs,rewards,done=step_envs(envs,actions);total+=rewards
        for env in envs:env.close()
        return dict(returns=total.tolist(),mean_return=float(total.mean()),steps=spec['evaluation_steps']*n,
            seconds=time.perf_counter()-start,planning_calls=planner.calls if planner else 0,planning_model_examples=planner.model_examples if planner else 0)
    def probe(learner,gravity,seed):
        if learner is None:return None
        envs,actual=envs_for(gravity,seed);imagined=actual.clone();g=torch.Generator().manual_seed(seed+700000);rows=[];real_total=torch.zeros(len(envs));pred_total=torch.zeros(len(envs));start=time.perf_counter()
        for t in range(spec['probe_steps']):
            actions=2*torch.rand(len(envs),generator=g)-1
            one_step=learner.predict(inputs(actual,actions));imagined,pred_reward=advance(learner,imagined,actions)
            following,rewards,_=step_envs(envs,actions);truth=targets(actual,following,rewards)
            rows.append(dict(horizon=t+1,one_step_target_mse=float((one_step-truth).square().mean()),
                imagined_state_mse=float((imagined-following).square().mean()),imagined_reward_mse=float(((pred_reward-torch.tensor(rewards))/16.).square().mean())))
            actual=following;real_total+=torch.tensor(rewards);pred_total+=pred_reward
        for env in envs:env.close()
        return dict(rows=rows,actual_returns=real_total.tolist(),predicted_returns=pred_total.tolist(),steps=spec['probe_steps']*len(envs),seconds=time.perf_counter()-start)
    records=[]
    for seed in spec['seeds']:
        for arm in spec['arms']:
            path=a.out/f'{seed}-{arm}.json';log_path=a.out/f'{seed}-{arm}-stages.jsonl'
            if a.execute and path.exists():r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
            if a.execute:assert not log_path.exists(),'Partial arm exists; inspect its checkpoint before restarting'
            torch.manual_seed(seed);learner=create_learner(arm,cfg);random_g=torch.Generator().manual_seed(seed+1000000)
            stages=[];lifetime_steps=0;training_steps=0;evaluation_steps=0;probe_steps=0;started=time.perf_counter()
            for stage,gravity in enumerate(spec['gravity_sequence']):
                envs,obs=envs_for(gravity,seed+100000*stage);n=len(envs);episode_returns=np.zeros(n);completed=[];losses=[];rows=[]
                planner=None if learner is None else Planner(learner,spec,n,seed+2000000+stage,1 if arm=='replay_mpc1' else spec['planner_horizon'])
                if not a.execute and learner is not None:
                    before=[p.detach().clone() for m in learner.models for p in m.parameters()];f=learner.forward_examples
                    planner.act(obs)
                    assert all(torch.equal(x,p) for x,p in zip(before,[p for m in learner.models for p in m.parameters()]))
                    assert learner.forward_examples-f==n*spec['planner_population']*planner.h*spec['planner_iterations']
                stage_start=time.perf_counter()
                for t in range(spec['steps_per_stage']):
                    actions=2*torch.rand(n,generator=random_g)-1 if learner is None or lifetime_steps<spec['initial_random_steps'] else planner.act(obs)
                    assert torch.isfinite(actions).all() and actions.abs().max()<=1
                    x=inputs(obs,actions);prediction=None if learner is None else learner.predict(x)
                    following,rewards,done=step_envs(envs,actions);y=targets(obs,following,rewards)
                    if learner is not None:
                        loss=float((prediction-y).square().mean());assert math.isfinite(loss);losses.append(loss);learner.observe(x,y)
                    lifetime_steps+=1;training_steps+=n;episode_returns+=rewards;obs=following
                    for i,ended in enumerate(done):
                        if ended:
                            completed.append(float(episode_returns[i]));episode_returns[i]=0;obs[i]=normalized([envs[i].reset()[0]])[0]
                            if planner:planner.reset([i])
                    if (t+1)%200==0 or t+1==spec['steps_per_stage']:
                        row=dict(vector_steps=t+1,environment_steps=(t+1)*n,completed_episode_returns=completed.copy(),
                            prequential_mse=float(np.mean(losses)) if losses else None,seconds=time.perf_counter()-stage_start,
                            costs=learner.costs() if learner else None,planning_calls=planner.calls if planner else 0,planning_model_examples=planner.model_examples if planner else 0)
                        rows.append(row)
                        if a.execute:print(json.dumps(dict(seed=seed,arm=arm,stage=stage,vector_steps=t+1,mean_return=float(np.mean(completed)) if completed else None,mse=row['prequential_mse'])),flush=True)
                if not a.execute and learner is not None:
                    assert any(not torch.equal(x,p) for x,p in zip(before,[p for m in learner.models for p in m.parameters()]))
                for env in envs:env.close()
                training_seconds=time.perf_counter()-stage_start
                with torch.random.fork_rng():
                    ev=evaluate(learner,arm,gravity,seed+3000000+10000*stage);pr=probe(learner,gravity,seed+4000000+10000*stage)
                evaluation_steps+=ev['steps'];probe_steps+=pr['steps'] if pr else 0
                row=dict(stage=stage,gravity=gravity,training=rows,training_returns=completed,training_seconds=training_seconds,evaluation=ev,probe=pr,costs=learner.costs() if learner else None)
                stages.append(row)
                if a.execute:
                    with log_path.open('a') as f:f.write(json.dumps(row)+'\n')
                    if learner is not None:torch.save(dict(identity=identity,learner=learner,stages=stages,training_steps=training_steps,lifetime_steps=lifetime_steps,torch_rng=torch.get_rng_state(),action_rng=random_g.get_state()),a.out/f'{seed}-{arm}-stage{stage}.pt')
                print(json.dumps(dict(seed=seed,arm=arm,stage=stage,evaluation_mean_return=ev['mean_return'])),flush=True)
            r=dict(seed=seed,arm=arm,identity=identity,stages=stages,training_environment_steps=training_steps,evaluation_environment_steps=evaluation_steps,probe_environment_steps=probe_steps,seconds=time.perf_counter()-started,costs=learner.costs() if learner else None)
            if a.execute:path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
    if a.execute:(a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    else:(a.out/'preflight.json').write_text(json.dumps(dict(identity=identity,checks=['real observed transitions and parameter updates','planner leaves weights unchanged','exact planning forward count','finite bounded actions','all five interfaces exercised'],records=records),indent=2)+'\n')
    print('E14_COMPLETE' if a.execute else 'E14 neural world-model/planner preflight passed',flush=True)
if __name__=='__main__':main()
