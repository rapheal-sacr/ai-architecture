"""Frozen E32 optimization controls and acquisition evaluation; no RSI claim."""
from pathlib import Path
import argparse,copy,gc,hashlib,json,subprocess,time,importlib.metadata
import numpy as np,torch,transformers
from delivery_training_data import make_data,assert_disjoint
from delivery_checkpoint_evaluation import tutorial_queries,closed_loop
from language_delivery_agent import LanguageActor
from language_training_controls import train_group
from preflight_language_adapter import base_hash,adapter_state
from e28_depth import state_hash
HERE=Path(__file__).parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run_case(seed,arm,cfg,scfg,data,model,out):
    start_all=time.perf_counter();actor=LanguageActor(model,scfg,seed);initial_adapter=state_hash(adapter_state(actor.model));base=base_hash(actor.model);actor.start_optimizer(arm['learning_rate']);trace=[];stages=[];last=0;training_seconds=0;opt_steps=0
    for milestone in cfg['milestones']:
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        for offset in range(last,milestone,arm['batch']):
            indices=data['order'][offset:offset+arm['batch']];assert len(indices)==arm['batch'];record=train_group(actor,[data['examples'][i] for i in indices]);opt_steps+=1;trace.append(dict(offset=offset,indices=indices,optimizer_step=opt_steps,**record))
        torch.cuda.synchronize();stage_seconds=time.perf_counter()-start;training_seconds+=stage_seconds;training_peak=torch.cuda.max_memory_allocated();assert actor.work['update_calls']==milestone;assert actor.work['inference_calls']==0;state=actor.state();state_before=state_hash(state);cp=out/f'{seed}-{arm["name"]}-{milestone}.pt';torch.save(dict(seed=seed,arm=arm,milestone=milestone,actor=state,config=cfg,source_config=scfg,data_hash=data['example_sha256'],order_hash=data['order_sha256']),cp);cp_hash=sha(cp)
        torch.cuda.reset_peak_memory_stats();diag=tutorial_queries(actor,dict(data,presented_order=data['order'][:milestone]));operational=closed_loop(actor,data,cfg);eval_peak=torch.cuda.max_memory_allocated();after=actor.state()
        for key in ('adapter','optimizer','replay','seen','learning_rng','torch_rng','cuda_rng'):assert state_hash(after[key])==state_hash(state[key])
        actor.load_state(state);assert state_hash(actor.state())==state_before;assert sha(cp)==cp_hash;stage=dict(milestone=milestone,stage_training_seconds=stage_seconds,cumulative_training_seconds=training_seconds,optimizer_steps=opt_steps,training_work=copy.deepcopy(actor.work),training_costs=actor.costs(),training_cuda_peak_bytes=training_peak,evaluation_cuda_peak_bytes=eval_peak,diagnostic=diag,closed_loop=operational,checkpoint=str(cp),checkpoint_sha256=cp_hash,state_sha256=state_before,evaluation_did_not_update_parameters=True,post_evaluation_training_state_restore_exact=True);stages.append(stage);(out/f'{seed}-{arm["name"]}-{milestone}.json').write_text(json.dumps(stage,indent=2)+'\n');print(json.dumps(dict(event='E32_STAGE',seed=seed,arm=arm['name'],milestone=milestone,tutorial_correct=sum(r['rule_correct'] for r in diag['rows']),tutorial_queries=len(diag['rows']),deliveries=operational['total_successes'],actions=operational['total_actions'],training_seconds=stage_seconds)),flush=True);last=milestone
    assert base_hash(actor.model)==base;result=dict(seed=seed,arm=arm,initial_adapter_sha256=initial_adapter,base_hash=base,base_unchanged=True,model_load_seconds=actor.load_seconds,total_case_seconds=time.perf_counter()-start_all,teacher_actions=data['teacher_actions'],data_hash=data['example_sha256'],order_hash=data['order_sha256'],trace=trace,stages=stages);(out/f'{seed}-{arm["name"]}.json').write_text(json.dumps(result,indent=2)+'\n');del actor;gc.collect();torch.cuda.empty_cache();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();cfg=json.loads((HERE/'protocol_e32.json').read_text());scfg=json.loads((HERE/cfg['source_protocol']).read_text());torch.set_num_threads(scfg['torch_threads']);torch.use_deterministic_algorithms(True)
    if a.preflight:cfg.update(seeds=[1911],milestones=[8,16],tutorial_worlds=1,eval_world_sizes=[2,3],eval_missions=[dict(world=0,target=0,budget=8),dict(world=1,target=1,budget=8)])
    assert sha(a.model/'model.safetensors')==scfg['base_weights_sha256'];source=json.loads((a.model/'source_identity.json').read_text());assert source['commit']==scfg['base_model_commit'];tr=Path(transformers.__file__).resolve().parents[2];trhead=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tr,text=True).strip();assert trhead==scfg['transformers_commit'];a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'complete.json').exists();files=['protocol_e32.json',cfg['source_protocol'],'e32_training_controls.py','language_training_controls.py','delivery_training_data.py','delivery_checkpoint_evaluation.py','delivery_diagnostics.py','delivery_world.py','language_delivery_agent.py','preflight_language_adapter.py','e28_depth.py'];manifest=dict(config=cfg,source_config=scfg,preflight=a.preflight,git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_identity={f:sha(HERE/f) for f in files},model_source=source,transformers_commit=trhead,versions={m:importlib.metadata.version(m) for m in ['torch','numpy','transformers','tokenizers','safetensors','huggingface-hub']});(a.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');start=time.perf_counter();datasets=[make_data(seed,cfg) for seed in cfg['seeds']];disjoint=assert_disjoint(datasets);generation_seconds=time.perf_counter()-start;dataset_files=[]
    for data in datasets:
        dp=a.out/f'{data["seed"]}-data.json';dp.write_text(json.dumps(data,indent=2)+'\n');dataset_files.append(dict(seed=data['seed'],path=str(dp),sha256=sha(dp)))
    (a.out/'data_manifest.json').write_text(json.dumps(dict(files=dataset_files,disjoint=disjoint,generation_seconds=generation_seconds),indent=2)+'\n');results=[];planners=[]
    for data in datasets:
        planner=dict(seed=data['seed'],**closed_loop(None,data,cfg));planners.append(planner);(a.out/f'{data["seed"]}-planner.json').write_text(json.dumps(planner,indent=2)+'\n')
        for arm in cfg['arms']:results.append(run_case(data['seed'],arm,cfg,scfg,data,a.model,a.out))
    for seed in cfg['seeds']:
        group=[r for r in results if r['seed']==seed];assert len({r['initial_adapter_sha256'] for r in group})==1
        for i,milestone in enumerate(cfg['milestones']):
            for key in ('update_calls','update_tokens','update_attention_elements'):assert len({r['stages'][i]['training_work'][key] for r in group})==1
            for mi in range(len(cfg['eval_missions'])):assert len({r['stages'][i]['closed_loop']['episodes'][mi]['world_definition_hash'] for r in group})==1
    complete=dict(status='PREFLIGHT_ONLY' if a.preflight else 'E32_COMPLETE',manifest=manifest,dataset_files=dataset_files,data_disjoint=disjoint,data_generation_seconds=generation_seconds,planners=planners,results=results);(a.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],cases=len(results),snapshots=sum(len(r['stages']) for r in results))),flush=True)
if __name__=='__main__':main()
