from pathlib import Path
import argparse,gc,hashlib,importlib.util,json,random,subprocess,time
import numpy as np
import torch
from torch.nn import functional as F
from torch.distributions import Categorical
from e8_closed_loop import SyncEnv

HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--repos',type=Path,required=True)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');a=p.parse_args()
    torch.set_num_threads(1);spec=json.loads((HERE/'protocol_e10.json').read_text());base_spec=json.loads((HERE/'protocol_e8.json').read_text())
    for repo,head in base_spec['commits'].items():
        assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.repos/repo,text=True).strip()==head
    import sys
    sys.path.insert(0,str(a.repos/'loss-of-plasticity'))
    import gymnasium as gym
    import minigrid,torch_ac
    import torch_ac.algos.base as base
    from lop.utils.AdamGnT import AdamGnT
    base.ParallelEnv=SyncEnv
    ms=importlib.util.spec_from_file_location('policy_source',a.repos/'rl-starter-files/model.py')
    source=importlib.util.module_from_spec(ms);ms.loader.exec_module(source)
    device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device)
    class Policy(source.ACModel):
        detached=False
        def forward(self,obs,memory):
            if not self.detached:return super().forward(obs,memory)
            x=self.image_conv(obs.image.transpose(1,3).transpose(2,3));x=x.reshape(x.shape[0],-1)
            hidden=self.memory_rnn(x,(memory[:,:self.semi_memory_size],memory[:,self.semi_memory_size:]))
            embedding=hidden[0];memory=torch.cat(hidden,dim=1)
            return Categorical(logits=F.log_softmax(self.actor(embedding),dim=1)),self.critic(embedding.detach()).squeeze(1),memory
    def preprocess(obss,device=None):
        return torch_ac.DictList(image=torch.tensor(np.array([o['image'] for o in obss]),dtype=torch.float32,device=device))
    def evaluate(model,name,seed,n):
        env=gym.make(name);rows=[];model.eval()
        with torch.random.fork_rng(devices=[device.index or 0] if device.type=='cuda' else []):
            torch.manual_seed(seed)
            with torch.no_grad():
                for i in range(n):
                    obs,_=env.reset(seed=seed+i);memory=torch.zeros(1,model.memory_size,device=device);ret=0.;steps=0
                    while True:
                        dist,_,memory=model(preprocess([obs],device),memory)
                        obs,r,done,trunc,_=env.step(int(dist.sample()));ret+=r;steps+=1
                        if done or trunc:break
                    rows.append(dict(seed=seed+i,returnn=ret,steps=steps,success=ret>0))
        env.close();model.train();return dict(environment=name,episodes=rows,success_rate=float(np.mean([r['success'] for r in rows])))
    def run_case(checkpoint,seed,arm):
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        envs=[]
        for i in range(16):
            e=gym.make(spec['environment']);e.reset(seed=seed+100000+10000*i);envs.append(e)
        model=Policy({'image':(7,7,3)},envs[0].action_space,use_memory=True,use_text=False).to(device)
        model.detached='detached' in arm
        if not a.execute:
            model.detached=True
            dummy=preprocess([envs[0].reset(seed=77)[0]],device)
            memory=torch.zeros(1,model.memory_size,device=device)
            model.zero_grad();dist,value,_=model(dummy,memory);value.sum().backward()
            assert all(p.grad is None for p in model.image_conv.parameters())
            assert any(p.grad is not None for p in model.critic.parameters())
            model.zero_grad();dist,_,_=model(dummy,memory);dist.log_prob(torch.zeros(1,dtype=torch.long,device=device)).sum().backward()
            assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.image_conv.parameters())
            model.zero_grad()
        # Locally generated E8 checkpoint, not an externally supplied pickle.
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e8.json','e8_closed_loop.py')}
        assert all(payload['identity'][n]==h for n,h in expected.items())
        model.load_state_dict(payload['model']);opt=AdamGnT(model.parameters(),lr=base_spec['learning_rate'])
        opt.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng']);np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(payload['cuda_rng'])
        counts={'mode':'rollout','rollout':0,'update':0,'evaluation':0,'optimizer_updates':0}
        def forward_hook(module,inputs):counts[counts['mode']]+=len(inputs[0].image)
        handle=model.register_forward_pre_hook(forward_hook)
        original_step=opt.step
        def step(*args,**kwargs):
            result=original_step(*args,**kwargs);counts['optimizer_updates']+=1;return result
        opt.step=step
        rows=[];frames=0;started=time.perf_counter()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
        if arm!='frozen_no_update':
            algo=torch_ac.PPOAlgo(envs,model,device,num_frames_per_proc=128,recurrence=4,epochs=4,batch_size=256,
                preprocess_obss=preprocess,lr=base_spec['learning_rate'],entropy_coef=0. if arm.startswith('no_entropy') else .01)
            algo.optimizer=opt
            while frames<(spec['frames'] if a.execute else 2048):
                counts['mode']='rollout';exps,logs=algo.collect_experiences();counts['mode']='update';metrics=algo.update_parameters(exps)
                frames+=logs['num_frames'];assert all(np.isfinite(v) for v in metrics.values())
                rows.append(dict(frames=frames,mean_return=float(np.mean(logs['return_per_episode'])),**{k:float(v) for k,v in metrics.items()}))
                if a.execute and len(rows)%32==0:print(json.dumps(dict(seed=seed,arm=arm,frames=frames,returnn=rows[-1]['mean_return'])),flush=True)
            algo.env.close()
        else:
            for env in envs:env.close()
        if device.type=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;peak=torch.cuda.max_memory_allocated() if device.type=='cuda' else None
        counts['mode']='evaluation'
        evals=[evaluate(model,en,seed+2000000+1000*j,spec['evaluation_episodes'] if a.execute else 2)
            for j,en in enumerate(dict.fromkeys(base_spec['environments']))]
        handle.remove()
        return dict(seed=seed,arm=arm,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            source_frames=payload['frames'],training_frames=frames,seconds=elapsed,training=rows,evaluations=evals,
            counts={k:v for k,v in counts.items() if k!='mode'},parameters=sum(p.numel() for p in model.parameters()),peak_gpu_bytes=peak)
    identity={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('protocol_e8.json','e8_closed_loop.py','protocol_e10.json','e10_objective.py')}
    identity.update(base_spec['commits']);a.out.mkdir(parents=True,exist_ok=True);records=[]
    checkpoints=spec['starting_checkpoints'] if a.execute else spec['starting_checkpoints'][:1]
    arms=spec['arms'] if a.execute else ['no_entropy_detached_critic']
    for relative in checkpoints:
        checkpoint=a.runs/relative;seed=int(checkpoint.name.split('-')[0])
        for arm in arms:
            path=a.out/f'{seed}-{arm}.json'
            if a.execute and path.exists():r=json.loads(path.read_text());assert r['identity']==identity
            else:
                r=run_case(checkpoint,seed,arm);r['identity']=identity
                if a.execute:path.write_text(json.dumps(r,indent=2)+'\n')
            records.append(r);print(json.dumps(dict(seed=seed,arm=arm,success=[e['success_rate'] for e in r['evaluations']])),flush=True)
    if a.execute:(a.out/'complete.json').write_text(json.dumps(dict(protocol=spec,identity=identity,records=records),indent=2)+'\n')
    print('E10_COMPLETE' if a.execute else 'E10 counterfactual preflight passed',flush=True)

if __name__=='__main__':main()
