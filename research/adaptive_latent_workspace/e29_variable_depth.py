"""E29 fixed versus balanced variable recurrence with exactly paired message work."""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np,torch
from clrs_reference import Reference
from reasoning_processor import ReasoningProcessor
from e27_reasoning import Reservoir,batch_tensors,sync,tensorbytes,digest,filehash,make_data,evaluate as base_evaluate
HERE=Path(__file__).parent

def depth_schedule(seed,arm,cfg):
    updates=cfg['updates_per_stage'];stages=len(cfg['stages'])
    if arm=='fixed16_replay':return [cfg['baseline_depth']]*(updates*stages)
    if arm=='fixed24_replay':return [cfg['mean_depth']]*(updates*stages)
    assert arm=='variable_replay' and updates%2==0
    rng=np.random.default_rng(seed+700000);all_steps=[]
    for _ in range(stages):
        first=rng.integers(cfg['variable_min'],cfg['variable_max']+1,size=(updates-2)//2);paired=np.concatenate([first,2*cfg['mean_depth']-first]);rng.shuffle(paired);steps=[cfg['mean_depth']]+list(map(int,paired))+[cfg['mean_depth']]
        assert min(steps)>=cfg['variable_min'] and max(steps)<=cfg['variable_max'] and sum(steps)==updates*cfg['mean_depth'];all_steps+=steps
    return all_steps

def evaluate_all(model,ev,arm,cfg,device):
    public=base_evaluate(model,ev,'size',cfg,device)
    fixed=base_evaluate(model,ev,'short',dict(cfg,short_steps=cfg['mean_depth']),device)
    double={}
    for key,cell in ev.items():double[key]=base_evaluate(model,{key:cell},'short',dict(cfg,short_steps=2*cell['w'].shape[1]),device)[key]
    return dict(public_N=public,fixed24=fixed,public_2N=double)

def run_case(seed,arm,cfg,data,ev,ident,out,device):
    torch.manual_seed(seed+300000);model=ReasoningProcessor(cfg['hidden']).to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate']);memory=Reservoir(cfg['replay_capacity_graphs'] if arm.endswith('replay') else 0,cfg['train_nodes'],seed+400000)
    initial=digest([p.detach().cpu().numpy() for p in model.parameters()]);stage_results=[];trace=[];total_train_seconds=0.;examples=0;replay_examples=0;messages=0;decoders=0;max_cuda=0
    schedule=depth_schedule(seed,arm,cfg);stage_results.append(dict(stage=-1,task=None,evaluation=evaluate_all(model,ev,arm,cfg,device)))
    b=cfg['current_batch'];upd=cfg['updates_per_stage'];position=0;eval_fingerprints=[]
    for stage,task in enumerate(cfg['stages']):
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        sync(device);start=time.perf_counter()
        for step in range(upd):
            steps=schedule[stage*upd+step]
            cur={k:data[k][position:position+b] for k in ('w','s','t','y')};assert np.all(cur['t']==task);old=memory.sample(cfg['replay_batch']);mixed=cur if old is None else {k:np.concatenate([cur[k],old[k]]) for k in cur};w,s,t,y=batch_tensors(mixed,device)
            opt.zero_grad(set_to_none=True);logits=model(w,s,t,steps);loss=torch.nn.functional.cross_entropy(logits.reshape(-1,cfg['train_nodes']),y.reshape(-1));assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip']);assert torch.isfinite(norm);opt.step()
            trace.append(float(loss.detach().cpu()));nb=len(s);examples+=nb;replay_examples+=nb-b;messages+=nb*cfg['train_nodes']**2*steps;decoders+=nb*cfg['train_nodes']**2
            if memory.capacity:memory.add(cur)
            position+=b
        sync(device);train_seconds=time.perf_counter()-start;total_train_seconds+=train_seconds
        if device=='cuda':max_cuda=max(max_cuda,torch.cuda.max_memory_allocated())
        # Evaluations cannot mutate learned state, optimizer, memory, or RNG.
        before=digest([p.detach().cpu().numpy() for p in model.parameters()]);rng_before=torch.get_rng_state().clone();cuda_before=torch.cuda.get_rng_state_all() if device=='cuda' else [];mem_before=memory.state();eval_result=evaluate_all(model,ev,arm,cfg,device)
        assert before==digest([p.detach().cpu().numpy() for p in model.parameters()]);assert torch.equal(rng_before,torch.get_rng_state());assert all(torch.equal(x,y) for x,y in zip(cuda_before,torch.cuda.get_rng_state_all() if device=='cuda' else []));assert mem_before['rng']==memory.state()['rng'];eval_fingerprints.append(before)
        checkpoint=dict(seed=seed,arm=arm,stage=stage,depth_schedule=schedule,position=position,model=model.state_dict(),optimizer=opt.state_dict(),memory=memory.state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else [],config=cfg,data_identity=ident,initial_model_sha256=initial,trained_graph_presentations=examples,replay_graph_presentations=replay_examples,message_candidates=messages,decoder_candidates=decoders)
        cp=out/f'{seed}-{arm}-stage{stage}.pt';torch.save(checkpoint,cp)
        stage_results.append(dict(stage=stage,task=task,train_seconds=train_seconds,mean_loss=float(np.mean(trace[-upd:])),evaluation=eval_result,checkpoint=str(cp),checkpoint_sha256=filehash(cp)))
        print(json.dumps(dict(event='E29_STAGE',seed=seed,arm=arm,stage=stage,train_seconds=train_seconds,random16={str(k):eval_result['public_N'][f'{k}-{cfg["train_nodes"]}-random']['functional_graph_accuracy'] for k in (0,1)})),flush=True)
    persistent=tensorbytes(model.state_dict())+tensorbytes(opt.state_dict())+tensorbytes(memory.state())+tensorbytes(torch.get_rng_state())+tensorbytes(torch.cuda.get_rng_state_all() if device=='cuda' else [])
    result=dict(seed=seed,arm=arm,source=ident,depth_schedule=schedule,recurrence_steps_total=sum(schedule),initial_model_sha256=initial,parameters=sum(p.numel() for p in model.parameters()),training_seconds=total_train_seconds,trained_graph_presentations=examples,current_graph_presentations=position,replay_graph_presentations=replay_examples,optimizer_steps=len(trace),message_candidates=messages,decoder_candidates=decoders,persistent_tensor_bytes=persistent,memory_allocated_array_bytes=sum(getattr(memory,k).nbytes for k in ('w','s','t','y')),cuda_peak_training_allocated_bytes=max_cuda,stages=stage_results,training_loss_trace=trace,evaluation_state_checks=eval_fingerprints)
    (out/f'{seed}-{arm}.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args();cfg=json.loads((HERE/'protocol_e29.json').read_text());torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True)
    if args.preflight:cfg.update(seeds=[1191],updates_per_stage=4,current_batch=4,replay_batch=4,replay_capacity_graphs=8,train_nodes=8,eval_nodes=[8,16],eval_graphs_per_task_size_family=2,baseline_depth=8,mean_depth=12,variable_min=8,variable_max=16)
    device=cfg['device'];assert torch.cuda.is_available();args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'complete.json').exists();ref=Reference(args.repo);assert ref.identity['graphs_sha256']=='75366fdd2bd72e8f84103dbda6417214fa071bd42237130909b0035b3cd34940'
    manifest=dict(config=cfg,preflight=args.preflight,code_identity={f:filehash(HERE/f) for f in ('protocol_e29.json','e29_variable_depth.py','e27_reasoning.py','reasoning_processor.py','clrs_reference.py')},git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),reference=ref.identity,torch=torch.__version__,numpy=np.__version__,device_name=torch.cuda.get_device_name(),cuda=torch.version.cuda)
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[]
    for seed in cfg['seeds']:
        data,ev,ident=make_data(seed,cfg,ref);torch.save(dict(train=data,evaluation=ev,identity=ident),args.out/f'{seed}-data.pt');print(json.dumps(dict(event='E29_DATA',seed=seed,identity=ident)),flush=True)
        for arm in cfg['arms']:results.append(run_case(seed,arm,cfg,data,ev,ident,args.out,device))
    audit=[]
    for result in results:
        saved=torch.load(result['stages'][-1]['checkpoint'],map_location=device,weights_only=False);model=ReasoningProcessor(cfg['hidden']).to(device);model.load_state_dict(saved['model']);ev=torch.load(args.out/f'{result["seed"]}-data.pt',weights_only=False)['evaluation'];reproduced=evaluate_all(model,ev,result['arm'],cfg,device)
        assert all(v['predictions']==result['stages'][-1]['evaluation'][policy][key]['predictions'] for policy,cells in reproduced.items() for key,v in cells.items());assert saved['position']==len(cfg['stages'])*cfg['updates_per_stage']*cfg['current_batch'];assert all(int(v['step'])==len(cfg['stages'])*cfg['updates_per_stage'] for v in saved['optimizer']['state'].values());assert all(torch.isfinite(p).all() for p in model.parameters());audit.append(dict(seed=result['seed'],arm=result['arm'],final_predictions_exact=True))
    pair_checks=[]
    for seed in cfg['seeds']:
        paired=[x for x in results if x['seed']==seed];assert len({x['initial_model_sha256'] for x in paired})==1
        for stage in range(len(cfg['stages'])):
            states=[torch.load(args.out/f'{seed}-{arm}-stage{stage}.pt',map_location='cpu',weights_only=False) for arm in cfg['arms']];mem=states[0]['memory']
            assert all(mem['rng']==x['memory']['rng'] and mem['seen']==x['memory']['seen'] and all(torch.equal(mem[k],x['memory'][k]) for k in ('w','s','t','y')) for x in states[1:]);assert states[1]['message_candidates']==states[2]['message_candidates'];assert len({x['trained_graph_presentations'] for x in states})==1;pair_checks.append(dict(seed=seed,stage=stage,all_replay_states_exact=True,variable_fixed24_message_counts_exact=True))
    complete=dict(manifest=manifest,results=results,audit=audit,pair_checks=pair_checks,status='PREFLIGHT_ONLY' if args.preflight else 'E29_COMPLETE');(args.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],cases=len(results),audit='passed')),flush=True)
if __name__=='__main__':main()
