from pathlib import Path
import argparse,hashlib,importlib.util,json,random,subprocess,time,gc
import numpy as np
import torch
from e8_closed_loop import SyncEnv
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repos',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    spec=json.loads((HERE/'protocol_e12.json').read_text());base_spec=json.loads((HERE/'protocol_e8.json').read_text())
    for repo,head in base_spec['commits'].items():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repos/repo,text=True).strip()==head
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=a.repos/repo,text=True).strip()
    import sys
    sys.path.insert(0,str(a.repos/'loss-of-plasticity'))
    import gymnasium as gym
    import minigrid,torch_ac
    import torch_ac.algos.base as base
    from lop.utils.AdamGnT import AdamGnT
    base.ParallelEnv=SyncEnv
    modspec=importlib.util.spec_from_file_location('policy_source',a.repos/'rl-starter-files/model.py')
    source=importlib.util.module_from_spec(modspec);modspec.loader.exec_module(source)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e8.json','e8_closed_loop.py','protocol_e12.json','e12_transfer.py')}
    identity.update(base_spec['commits']);a.out.mkdir(parents=True,exist_ok=True)
    def preprocess(obss,device=None):
        return torch_ac.DictList(image=torch.tensor(np.array([o['image'] for o in obss]),dtype=torch.float32,device=device))
    def envs_for(name,seed,n=16):
        envs=[]
        for i in range(n):
            e=gym.make(name);e.reset(seed=seed+10000*i);envs.append(e)
        return envs
    def evaluate(model,name,seed,n):
        env=gym.make(name);rows=[];model.eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            with torch.no_grad():
                for i in range(n):
                    obs,_=env.reset(seed=seed+i);memory=torch.zeros(1,model.memory_size,device=device);total=0.;steps=0
                    while True:
                        dist,_,memory=model(preprocess([obs],device),memory)
                        obs,r,done,trunc,_=env.step(int(dist.sample()));total+=r;steps+=1
                        if done or trunc:break
                    rows.append(dict(seed=seed+i,success=total>0,returnn=total,steps=steps))
        max_steps=env.unwrapped.max_steps;env.close();model.train()
        return dict(environment=name,success_rate=float(np.mean([r['success'] for r in rows])),episodes=rows,max_steps=max_steps)
    records=[]
    for seed in spec['seeds'] if a.execute else [77]:
        for arm,entropy in spec['arms'].items():
            result_path=a.out/f'{seed}-{arm}.json'
            if a.execute and result_path.exists():
                r=json.loads(result_path.read_text());assert r['identity']==identity;records.append(r);continue
            gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
            torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
            temp=gym.make(spec['environments'][0]);model=source.ACModel({'image':(7,7,3)},temp.action_space,use_memory=True,use_text=False).to(device);temp.close()
            opt=AdamGnT(model.parameters(),lr=base_spec['learning_rate'])
            counts=dict(mode='rollout',rollout=0,update=0,evaluation=0,optimizer_updates=0)
            def forward_hook(module,inputs):counts[counts['mode']]+=len(inputs[0].image)
            handle=model.register_forward_pre_hook(forward_hook);original_step=opt.step
            def step(*args,**kwargs):
                result=original_step(*args,**kwargs);counts['optimizer_updates']+=1;return result
            opt.step=step;stages=[];total_frames=0
            names=spec['environments'] if a.execute else spec['environments'][:1]
            for stage,name in enumerate(names):
                checkpoint=a.out/f'{seed}-{arm}-stage{stage}.pt'
                if a.execute and checkpoint.exists():
                    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);assert payload['identity']==identity
                    model.load_state_dict(payload['model']);opt.load_state_dict(payload['optimizer'])
                    for parameter,state in opt.state.items():
                        for key,value in state.items():
                            if isinstance(value,torch.Tensor):state[key]=value.to(parameter.device)
                    counts.update(payload['counts']);stages=payload['stages'];total_frames=payload['total_frames']
                    torch.set_rng_state(payload['torch_rng']);np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
                    if device.type=='cuda':torch.cuda.set_rng_state_all(payload['cuda_rng'])
                    continue
                envs=envs_for(name,seed+100000*stage);max_steps=envs[0].unwrapped.max_steps
                algo=torch_ac.PPOAlgo(envs,model,device,num_frames_per_proc=128,recurrence=4,epochs=4,batch_size=256,
                    preprocess_obss=preprocess,lr=base_spec['learning_rate'],entropy_coef=entropy)
                algo.optimizer=opt;rows=[];frames=0;start=time.perf_counter()
                if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
                while frames<(spec['frames_per_stage'] if a.execute else 2048):
                    counts['mode']='rollout';exps,logs=algo.collect_experiences();counts['mode']='update';metrics=algo.update_parameters(exps)
                    frames+=logs['num_frames'];total_frames+=logs['num_frames'];assert all(np.isfinite(v) for v in metrics.values())
                    row=dict(frames=frames,mean_return=float(np.mean(logs['return_per_episode'])),**{k:float(v) for k,v in metrics.items()})
                    rows.append(row)
                    if a.execute and len(rows)%32==0:print(json.dumps(dict(seed=seed,arm=arm,stage=stage,**row)),flush=True)
                if device.type=='cuda':torch.cuda.synchronize()
                elapsed=time.perf_counter()-start;peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None;algo.env.close()
                counts['mode']='evaluation';eval_start=time.perf_counter()
                evals=[evaluate(model,en,seed+2000000+1000*j,spec['evaluation_episodes'] if a.execute else 1) for j,en in enumerate(dict.fromkeys(spec['environments']))]
                row=dict(stage=stage,environment=name,max_steps=max_steps,frames=frames,training=rows,training_seconds=elapsed,
                    evaluations=evals,evaluation_seconds=time.perf_counter()-eval_start,counts=counts.copy(),peak_gpu_bytes=peak,
                    tensor_bytes=sum(p.numel()*p.element_size() for p in model.parameters())+sum(v.numel()*v.element_size() for s in opt.state.values() for v in s.values() if isinstance(v,torch.Tensor)))
                stages.append(row)
                if a.execute:
                    torch.save(dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),counts=counts.copy(),stages=stages,total_frames=total_frames,
                        torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None),checkpoint)
                print(json.dumps(dict(seed=seed,arm=arm,stage=stage,success=[e['success_rate'] for e in evals])),flush=True)
            handle.remove();r=dict(seed=seed,arm=arm,identity=identity,stages=stages,total_frames=total_frames,parameters=sum(p.numel() for p in model.parameters()))
            if a.execute:result_path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r)
    if a.execute:(a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E12_COMPLETE' if a.execute else 'E12 preflight passed',flush=True)
if __name__=='__main__':main()
