from pathlib import Path
import argparse,hashlib,importlib.util,json,random,subprocess,sys,time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent

class SyncEnv:
    def __init__(self,envs):self.envs=envs
    def reset(self):return [e.reset()[0] for e in self.envs]
    def step(self,actions):
        results=[]
        for env,action in zip(self.envs,actions):
            obs,reward,terminated,truncated,info=env.step(int(action))
            if terminated or truncated:obs,_=env.reset()
            results.append((obs,reward,terminated,truncated,info))
        return zip(*results)
    def close(self):
        for env in self.envs:env.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repos',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e8.json').read_text())
    for repo,head in spec['commits'].items():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repos/repo,text=True).strip()==head
    sys.path.insert(0,str(a.repos/'loss-of-plasticity'))
    import gymnasium as gym
    import minigrid,torch_ac
    import torch_ac.algos.base as base
    from lop.utils.AdamGnT import AdamGnT
    from lop.algos.gnt import GnT
    base.ParallelEnv=SyncEnv
    module_spec=importlib.util.spec_from_file_location('pinned_policy',a.repos/'rl-starter-files/model.py')
    policy=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(policy)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def preprocess(obss,device=None):
        return torch_ac.DictList(image=torch.tensor(np.array([o['image'] for o in obss]),dtype=torch.float32,device=device))
    def envs_for(name,seed,n):
        envs=[]
        for i in range(n):
            e=gym.make(name);e.reset(seed=seed+10000*i);envs.append(e)
        return envs
    def evaluate(model,name,seed,n):
        rows=[];env=gym.make(name);model.eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            with torch.no_grad():
                for i in range(n):
                    obs,_=env.reset(seed=seed+i);memory=torch.zeros(1,model.memory_size,device=device);steps=0;total=0.
                    while True:
                        dist,_,memory=model(preprocess([obs],device),memory)
                        obs,reward,done,truncated,_=env.step(int(dist.sample().item()));steps+=1;total+=reward
                        if done or truncated:break
                    rows.append(dict(seed=seed+i,success=total>0,steps=steps,returnn=total))
        model.train();env.close();return dict(environment=name,episodes=rows,
            success_rate=float(np.mean([r['success'] for r in rows])),mean_return=float(np.mean([r['returnn'] for r in rows])),
            mean_steps=float(np.mean([r['steps'] for r in rows])))
    a.out.mkdir(parents=True,exist_ok=True)
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e8.json','e8_closed_loop.py')}
    identity.update(spec['commits']);records=[]
    seeds=spec['seeds'] if a.execute else [77]
    arms=spec['arms'] if a.execute else ['recurrent_ppo_renewal']
    for seed in seeds:
        for arm in arms:
            path=a.out/f'{seed}-{arm}.json'
            if a.execute and path.exists():
                r=json.loads(path.read_text());assert r['identity']==identity;records.append(r);continue
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            envs=envs_for(spec['environments'][0],seed,16)
            model=policy.ACModel({'image':(7,7,3)},envs[0].action_space,use_memory=True,use_text=False).to(device)
            opt=AdamGnT(model.parameters(),lr=spec['learning_rate'])
            state={'updates':0,'replacements':0,'mode':'rollout','rollout':0,'update':0,'evaluation':0};features={}
            handles=[]
            def count_forward(module,inputs):state[state['mode']]+=len(inputs[0].image)
            handles.append(model.register_forward_pre_hook(count_forward))
            for name in ('actor','critic'):
                handles.append(getattr(model,name)[1].register_forward_hook(lambda module,inp,out,n=name:features.__setitem__(n,out.detach())))
            gn=[]
            if arm.endswith('renewal'):
                for name in ('actor','critic'):
                    gn.append((name,GnT(getattr(model,name),'tanh',opt,replacement_rate=spec['renewal_rate'] if a.execute else .1,
                        decay_rate=spec['renewal_decay'],maturity_threshold=spec['renewal_maturity'] if a.execute else 0,
                        util_type='adaptable_contribution',device=str(device))))
            original_step=opt.step
            def step(*args,**kwargs):
                result=original_step(*args,**kwargs);state['updates']+=1
                for name,g in gn:
                    g.gen_and_test([features[name]])
                    state['replacements']+=sum(int((age==0).sum()) for age in g.ages)
                return result
            opt.step=step
            def tensor_bytes():
                n=sum(p.numel()*p.element_size() for p in model.parameters())
                n+=sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor))
                n+=sum(v.numel()*v.element_size() for _,g in gn for name in ('util','bias_corrected_util','ages','mean_feature_act') for v in getattr(g,name))
                return n
            stages=[];total_frames=0
            stage_names=spec['environments'] if a.execute else spec['environments'][:1]
            for stage,name in enumerate(stage_names):
                if stage:envs=envs_for(name,seed+100000*stage,16)
                algo=torch_ac.PPOAlgo(envs,model,device,num_frames_per_proc=128,recurrence=spec['recurrence'],
                    epochs=spec['epochs'],batch_size=spec['batch_size'],preprocess_obss=preprocess,lr=spec['learning_rate'])
                algo.optimizer=opt;rows=[];frames=0;started=time.perf_counter()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                while frames<(spec['frames_per_stage'] if a.execute else 2048):
                    state['mode']='rollout';exps,logs=algo.collect_experiences()
                    state['mode']='update';update=algo.update_parameters(exps)
                    frames+=logs['num_frames'];total_frames+=logs['num_frames']
                    assert all(np.isfinite(v) for v in update.values())
                    rows.append(dict(frames=frames,mean_return=float(np.mean(logs['return_per_episode'])),**{k:float(v) for k,v in update.items()}))
                    if a.execute and len(rows)%32==0:
                        print(json.dumps(dict(seed=seed,arm=arm,stage=stage,frames=frames,mean_return=rows[-1]['mean_return'])),flush=True)
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-started;algo.env.close()
                peak_gpu_bytes=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
                state['mode']='evaluation'
                evals=[evaluate(model,en,seed+2000000+1000*j,spec['evaluation_episodes'] if a.execute else 2)
                       for j,en in enumerate(dict.fromkeys(spec['environments']))]
                stages.append(dict(stage=stage,environment=name,frames=frames,seconds=elapsed,evaluations=evals,training=rows,
                    optimizer_updates=state['updates'],feature_replacements=state['replacements'],tensor_bytes=tensor_bytes(),
                    peak_gpu_allocated_bytes=peak_gpu_bytes,forward_examples={k:state[k] for k in ('rollout','update','evaluation')}))
                if a.execute:
                    renewal_state={name:{k:getattr(g,k) for k in ('util','bias_corrected_util','ages','mean_feature_act','accumulated_num_features_to_replace')} for name,g in gn}
                    torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),renewal=renewal_state,counters=state.copy(),
                        frames=total_frames,identity=identity,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),
                        python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),
                        a.out/f'{seed}-{arm}-stage{stage}.pt')
                    print(json.dumps(dict(seed=seed,arm=arm,stage=stage,seconds=elapsed,success=[e['success_rate'] for e in evals])),flush=True)
            assert state['updates']>0
            if not a.execute:assert state['replacements']>0
            r=dict(seed=seed,arm=arm,identity=identity,device=str(device),stages=stages,total_frames=total_frames,
                parameters=sum(p.numel() for p in model.parameters()))
            if a.execute:path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
            for handle in handles:handle.remove()
    if a.execute:(a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E8_COMPLETE' if a.execute else 'E8 closed-loop/renewal preflight passed',flush=True)

if __name__=='__main__':main()
