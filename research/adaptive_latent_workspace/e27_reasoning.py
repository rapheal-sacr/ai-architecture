"""E27: registered supervised graph-reasoning pilot, not an integrated agent."""
from pathlib import Path
import argparse,copy,hashlib,json,os,subprocess,time
import numpy as np
import torch
from clrs_reference import Reference,ALGORITHMS,validate_tree
from reasoning_processor import ReasoningProcessor
HERE=Path(__file__).parent

def digest(x):
    h=hashlib.sha256()
    for a in x:h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
def filehash(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def tensorbytes(obj):
    if torch.is_tensor(obj):return obj.numel()*obj.element_size()
    if isinstance(obj,dict):return sum(tensorbytes(v) for v in obj.values())
    if isinstance(obj,(list,tuple)):return sum(tensorbytes(v) for v in obj)
    return 0

def graph(rng,n,family,p=.25,endpoint=False):
    weights=np.zeros((n,n),np.float32)
    for i in range(n-1):weights[i,i+1]=weights[i+1,i]=rng.uniform(.1,1.)
    if family=='random':
        for i in range(n):
            for j in range(i+2,n):
                if rng.random()<p:weights[i,j]=weights[j,i]=rng.uniform(.1,1.)
    source=int(rng.choice([0,n-1])) if endpoint else int(rng.integers(n))
    perm=rng.permutation(n);source=int(np.where(perm==source)[0][0]);return weights[perm][:,perm],source

def make_data(seed,cfg,ref):
    start=time.perf_counter();rng=np.random.default_rng(seed+100000);w=[];s=[];t=[];y=[];labels_sec=0.
    for task in cfg['stages']:
        for _ in range(cfg['updates_per_stage']*cfg['current_batch']):
            a,src=graph(rng,cfg['train_nodes'],'chain' if rng.random()<cfg['train_chain_probability'] else 'random',cfg['random_extra_edge_probability'])
            ts=time.perf_counter();pi,_=ref(ALGORITHMS[task],a,src);labels_sec+=time.perf_counter()-ts
            w.append(a);s.append(src);t.append(task);y.append(pi)
    data=dict(w=np.array(w),s=np.array(s,np.int64),t=np.array(t,np.int64),y=np.array(y,np.int64))
    train_sec=time.perf_counter()-start;ev={};rng=np.random.default_rng(seed+200000);start=time.perf_counter();eval_reference_sec=0.
    for n in cfg['eval_nodes']:
        for family in cfg['eval_families']:
            weights=[];sources=[]
            for _ in range(cfg['eval_graphs_per_task_size_family']):
                a,src=graph(rng,n,family,cfg['random_extra_edge_probability'],endpoint=family=='chain');weights.append(a);sources.append(src)
            for task in range(2):
                parents=[];exact_seconds=0.;ts=time.perf_counter()
                for a,src in zip(weights,sources):
                    query_start=time.perf_counter();pi,_=ref(ALGORITHMS[task],a,src);exact_seconds+=time.perf_counter()-query_start;validate_tree(a,src,pi,ALGORITHMS[task]);parents.append(pi)
                baseline=time.perf_counter()-ts;eval_reference_sec+=baseline
                ev[f'{task}-{n}-{family}']=dict(w=np.array(weights),s=np.array(sources,np.int64),t=np.full(len(weights),task,np.int64),y=np.array(parents,np.int64),reference_and_validation_seconds=baseline,exact_reference_query_seconds=exact_seconds)
    ident=dict(train_sha256=digest([data[k] for k in ('w','s','t','y')]),eval_sha256={k:digest([v[x] for x in ('w','s','t','y')]) for k,v in ev.items()},train_graphs=len(w),train_generation_seconds=train_sec,train_reference_seconds=labels_sec,eval_generation_and_validation_seconds=time.perf_counter()-start,eval_reference_and_validation_seconds=eval_reference_sec)
    return data,ev,ident

class Reservoir:
    def __init__(self,capacity,n,seed):
        self.capacity=capacity;self.size=0;self.seen=0;self.rng=np.random.default_rng(seed)
        self.w=np.empty((capacity,n,n),np.float32);self.s=np.empty(capacity,np.int64);self.t=np.empty(capacity,np.int64);self.y=np.empty((capacity,n),np.int64)
    def sample(self,batch):
        if self.size==0:return None
        ids=self.rng.integers(self.size,size=batch);return {k:getattr(self,k)[ids] for k in ('w','s','t','y')}
    def add(self,data):
        for i in range(len(data['s'])):
            self.seen+=1;j=self.size if self.size<self.capacity else int(self.rng.integers(self.seen))
            if j<self.capacity:
                for k in ('w','s','t','y'):getattr(self,k)[j]=data[k][i]
                self.size=min(self.capacity,self.size+1)
    def state(self):
        return dict(capacity=self.capacity,size=self.size,seen=self.seen,rng=copy.deepcopy(self.rng.bit_generator.state),**{k:torch.from_numpy(getattr(self,k)[:self.size].copy()) for k in ('w','s','t','y')})

def batch_tensors(data,device):return [torch.as_tensor(data[k],device=device) for k in ('w','s','t','y')]
def structure_valid(a,src,pi):
    if int(pi[src])!=src:return False
    for v in range(len(a)):
        seen=set();u=v
        while u!=src:
            if u in seen:return False
            seen.add(u);p=int(pi[u])
            if p<0 or p>=len(a) or a[p,u]<=0:return False
            u=p
    return True

@torch.no_grad()
def evaluate(model,ev,arm,cfg,device):
    model.eval();out={}
    for key,d in ev.items():
        n=d['w'].shape[1];steps=cfg['short_steps'] if arm.startswith('short') else n;pred=[];sync(device);start=time.perf_counter()
        for lo in range(0,len(d['s']),cfg['eval_batch']):
            w,s,t,y=batch_tensors({k:d[k][lo:lo+cfg['eval_batch']] for k in ('w','s','t','y')},device);pred.append(model(w,s,t,steps).argmax(-1).cpu().numpy())
        sync(device);seconds=time.perf_counter()-start;pred=np.concatenate(pred);valid=[];optimal=[];start=time.perf_counter()
        for a,src,pi,task in zip(d['w'],d['s'],pred,d['t']):
            ok=structure_valid(a,int(src),pi);valid.append(ok)
            if not ok:optimal.append(False);continue
            try:validate_tree(a,int(src),pi,ALGORITHMS[int(task)]);optimal.append(True)
            except AssertionError:optimal.append(False)
        equal=pred==d['y'];out[key]=dict(parent_accuracy=float(equal.mean()),canonical_graph_accuracy=float(equal.all(1).mean()),functional_graph_accuracy=float(np.mean(optimal)),invalid_tree_fraction=float(1-np.mean(valid)),neural_query_seconds=seconds,validation_seconds=time.perf_counter()-start,graphs=len(pred),steps=steps,message_candidates=int(len(pred)*n*n*steps),decoder_candidates=int(len(pred)*n*n),predictions=pred.tolist(),reference_and_validation_seconds=d['reference_and_validation_seconds'],exact_reference_query_seconds=d['exact_reference_query_seconds'])
    model.train();return out

def run_case(seed,arm,cfg,data,ev,ident,out,device):
    torch.manual_seed(seed+300000);model=ReasoningProcessor(cfg['hidden']).to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);memory=Reservoir(cfg['replay_capacity_graphs'] if arm.endswith('replay') else 0,cfg['train_nodes'],seed+400000)
    initial=digest([p.detach().cpu().numpy() for p in model.parameters()]);stage_results=[];trace=[];total_train_seconds=0.;examples=0;replay_examples=0;messages=0;decoders=0;max_cuda=0
    steps=cfg['short_steps'] if arm.startswith('short') else cfg['train_nodes'];stage_results.append(dict(stage=-1,task=None,evaluation=evaluate(model,ev,arm,cfg,device)))
    b=cfg['current_batch'];upd=cfg['updates_per_stage'];position=0;eval_fingerprints=[]
    for stage,task in enumerate(cfg['stages']):
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        sync(device);start=time.perf_counter()
        for step in range(upd):
            cur={k:data[k][position:position+b] for k in ('w','s','t','y')};assert np.all(cur['t']==task);old=memory.sample(cfg['replay_batch']);mixed=cur if old is None else {k:np.concatenate([cur[k],old[k]]) for k in cur};w,s,t,y=batch_tensors(mixed,device)
            opt.zero_grad(set_to_none=True);logits=model(w,s,t,steps);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,cfg['train_nodes']),y.reshape(-1));assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm);opt.step()
            trace.append(float(loss.detach().cpu()));nb=len(s);examples+=nb;replay_examples+=nb-b;messages+=nb*cfg['train_nodes']**2*steps;decoders+=nb*cfg['train_nodes']**2
            if memory.capacity:memory.add(cur)
            position+=b
        sync(device);train_seconds=time.perf_counter()-start;total_train_seconds+=train_seconds
        if device=='cuda':max_cuda=max(max_cuda,torch.cuda.max_memory_allocated())
        # Evaluations cannot mutate learned state, optimizer, memory, or RNG.
        before=digest([p.detach().cpu().numpy() for p in model.parameters()]);rng_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state_all() if device=='cuda' else [];mem_before=memory.state();eval_result=evaluate(model,ev,arm,cfg,device)
        assert before==digest([p.detach().cpu().numpy() for p in model.parameters()]);assert torch.equal(rng_before,torch.get_rng_state());assert all(torch.equal(x,y) for x,y in zip(cuda_before,torch.cuda.get_rng_state_all() if device=='cuda' else []));assert mem_before['rng']==memory.state()['rng'];eval_fingerprints.append(before)
        checkpoint=dict(seed=seed,arm=arm,stage=stage,position=position,model=model.state_dict(),optimizer=opt.state_dict(),memory=memory.state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],config=cfg,data_identity=ident,initial_model_sha256=initial,trained_graph_presentations=examples,replay_graph_presentations=replay_examples,message_candidates=messages,decoder_candidates=decoders)
        cp=out/f'{seed}-{arm}-stage{stage}.pt';torch.save(checkpoint,cp)
        stage_results.append(dict(stage=stage,task=task,train_seconds=train_seconds,mean_loss=float(np.mean(trace[-upd:])),evaluation=eval_result,checkpoint=str(cp),checkpoint_sha256=filehash(cp)))
        print(json.dumps(dict(event='E27_STAGE',seed=seed,arm=arm,stage=stage,train_seconds=train_seconds,random16={str(k):eval_result[f'{k}-{cfg["train_nodes"]}-random']['functional_graph_accuracy'] for k in (0,1)})),flush=True)
    persistent=tensorbytes(model.state_dict())+tensorbytes(opt.state_dict())+tensorbytes(memory.state())+tensorbytes(torch.get_rng_state())+tensorbytes(torch.cuda.get_rng_state_all() if device=='cuda' else [])
    result=dict(seed=seed,arm=arm,source=ident,initial_model_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),training_seconds=total_train_seconds,trained_graph_presentations=examples,current_graph_presentations=position,replay_graph_presentations=replay_examples,optimizer_steps=len(trace),message_candidates=messages,decoder_candidates=decoders,persistent_tensor_bytes=persistent,memory_allocated_array_bytes=sum(getattr(memory,k).nbytes for k in ('w','s','t','y')),cuda_peak_training_allocated_bytes=max_cuda,stages=stage_results,training_loss_trace=trace,evaluation_state_checks=eval_fingerprints)
    (out/f'{seed}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args();cfg=json.loads((HERE/'protocol_e27.json').read_text());torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    if args.preflight:cfg.update(seeds=[1171],updates_per_stage=2,current_batch=4,replay_batch=4,replay_capacity_graphs=8,train_nodes=8,eval_nodes=[8,16],eval_graphs_per_task_size_family=2)
    device=cfg['device'];assert device!='cuda' or torch.cuda.is_available();args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'complete.json').exists(),'Refuse to overwrite completed run'
    ref=Reference(args.repo);assert ref.identity['graphs_sha256']=='75366fdd2bd72e8f84103dbda6417214fa071bd42237130909b0035b3cd34940'
    manifest=dict(config=cfg,preflight=args.preflight,code_identity={f:filehash(HERE/f) for f in ('protocol_e27.json','e27_reasoning.py','reasoning_processor.py','clrs_reference.py')},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),reference=ref.identity,torch=torch.__version__,numpy=np.__version__,device_name=torch.cuda.get_device_name() if device=='cuda' else 'cpu',cuda=torch.version.cuda)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    for seed in cfg['seeds']:
        data,ev,ident=make_data(seed,cfg,ref);torch.save(dict(train=data,evaluation=ev,identity=ident),args.out/f'{seed}-data.pt');print(json.dumps(dict(event='E27_DATA',seed=seed,identity=ident)),flush=True)
        for arm in cfg['arms']:results.append(run_case(seed,arm,cfg,data,ev,ident,args.out,device))
    # A saved model must reproduce each saved final evaluation prediction exactly.
    audit=[]
    for result in results:
        saved=torch.load(result['stages'][-1]['checkpoint'],map_location=device,weights_only=False);model=ReasoningProcessor(cfg['hidden']).to(device);model.load_state_dict(saved['model']);ev=torch.load(args.out/f'{result["seed"]}-data.pt',weights_only=False)['evaluation'];reproduced=evaluate(model,ev,result['arm'],cfg,device)
        assert all(v['predictions']==result['stages'][-1]['evaluation'][k]['predictions'] for k,v in reproduced.items());assert saved['position']==len(cfg['stages'])*cfg['updates_per_stage']*cfg['current_batch'];assert all(int(v['step'])==len(cfg['stages'])*cfg['updates_per_stage'] for v in saved['optimizer']['state'].values());assert all(torch.isfinite(p).all() for p in model.parameters());audit.append(dict(seed=result['seed'],arm=result['arm'],final_predictions_exact=True))
    for seed in cfg['seeds']:
        paired=[x for x in results if x['seed']==seed];assert len({x['initial_model_sha256'] for x in paired})==1
        a=torch.load(args.out/f'{seed}-short_replay-stage2.pt',weights_only=False)['memory'];b=torch.load(args.out/f'{seed}-size_replay-stage2.pt',weights_only=False)['memory'];assert a['rng']==b['rng'] and a['seen']==b['seen'] and all(torch.equal(a[k],b[k]) for k in ('w','s','t','y'))
    complete=dict(manifest=manifest,results=results,audit=audit,status='PREFLIGHT_ONLY' if args.preflight else 'E27_COMPLETE');(args.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],cases=len(results),audit='passed')),flush=True)
if __name__=='__main__':main()
