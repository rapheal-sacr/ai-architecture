"""Frozen inference-depth test; E27 sources and predictions remain immutable."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time,copy
import numpy as np
import torch
from reasoning_processor import ReasoningProcessor
from e27_reasoning import evaluate,filehash,digest
HERE=Path(__file__).parent

def state_hash(obj):
    h=hashlib.sha256()
    def add(v):
        h.update(type(v).__name__.encode())
        if torch.is_tensor(v):h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,np.ndarray):h.update(str(v.dtype).encode());h.update(str(v.shape).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=lambda x:repr(x)):add(k);add(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:add(x)
        else:h.update(repr(v).encode())
    add(obj);return h.hexdigest()

def run_source(source,source_data,cfg,source_cfg,out,preflight):
    cp_path=Path(source['stages'][-1]['checkpoint']);assert filehash(cp_path)==source['stages'][-1]['checkpoint_sha256'];saved=torch.load(cp_path,map_location='cpu',weights_only=False);before=state_hash(saved);model=ReasoningProcessor(source_cfg['hidden']).to(cfg['device']);model.load_state_dict(saved['model']);rng=torch.get_rng_state().clone();cuda=[x.clone() for x in torch.cuda.get_rng_state_all()];grid={};audit=[];start=time.perf_counter()
    for key,cell in source_data['evaluation'].items():
        n=cell['w'].shape[1];grid[key]={};assert source_data['identity']['eval_sha256'][key]==digest([cell[k] for k in ('w','s','t','y')])
        for depth in cfg['fixed_depths']:
            ecfg=dict(source_cfg,short_steps=depth,eval_batch=cfg['eval_batch']);row=evaluate(model,{key:cell},'short_diagnostic',ecfg,cfg['device'])[key];grid[key][str(depth)]=row
        original_depth=source_cfg['short_steps'] if source['arm'].startswith('short') else n
        assert grid[key][str(original_depth)]['predictions']==source['stages'][-1]['evaluation'][key]['predictions'];audit.append(dict(cell=key,source_predictions_exact=True,original_depth=original_depth))
    assert all(torch.equal(p.cpu(),saved['model'][k]) for k,p in model.state_dict().items());assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all()));assert before==state_hash(saved);assert filehash(cp_path)==source['stages'][-1]['checkpoint_sha256']
    policies={}
    for policy in cfg['primary_policies']:
        policies[policy]={}
        for key,rows in grid.items():
            n=source_data['evaluation'][key]['w'].shape[1]
            depth=(source_cfg['short_steps'] if source['arm'].startswith('short') else source_cfg['train_nodes']) if policy=='training_depth' else (16 if policy=='fixed16' else n if policy=='public_N' else 2*n)
            if str(depth) in rows:policies[policy][key]=dict(depth=depth,**{k:v for k,v in rows[str(depth)].items() if k!='predictions'})
            else:assert preflight
    result=dict(seed=source['seed'],source_arm=source['arm'],source_checkpoint=str(cp_path),source_checkpoint_sha256=source['stages'][-1]['checkpoint_sha256'],loaded_state_sha256=before,grid=grid,policies=policies,source_restore_checks=audit,weights_optimizer_memory_rng_unchanged=True,diagnostic_total_seconds=time.perf_counter()-start,executed_query_seconds=sum(v['neural_query_seconds'] for rows in grid.values() for v in rows.values()),executed_message_candidates=sum(v['message_candidates'] for rows in grid.values() for v in rows.values()),executed_decoder_candidates=sum(v['decoder_candidates'] for rows in grid.values() for v in rows.values()),executed_graph_queries=sum(v['graphs'] for rows in grid.values() for v in rows.values()))
    (out/f'{source["seed"]}-{source["arm"]}.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(event='E28_SOURCE_COMPLETE',seed=source['seed'],arm=source['arm'],query_seconds=result['executed_query_seconds'],audit='passed')),flush=True);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight',action='store_true');args=p.parse_args();cfg=json.loads((HERE/'protocol_e28.json').read_text());source_path=args.source/'complete.json';source=json.loads(source_path.read_text());source_hash=filehash(source_path);scfg=source['manifest']['config']
    if args.preflight:assert source['status']=='PREFLIGHT_ONLY';cfg.update(seeds=scfg['seeds'],fixed_depths=[2,8,16,32])
    else:assert source['status']=='E27_COMPLETE' and source['manifest']['git_commit']==cfg['source_commit'];assert scfg['seeds']==cfg['seeds']
    assert scfg['arms']==cfg['source_arms'];torch.set_num_threads(cfg['torch_threads']);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available();args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'complete.json').exists()
    manifest=dict(config=cfg,preflight=args.preflight,source_complete_sha256=source_hash,source_path=str(source_path),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=HERE,text=True).strip(),code_identity={f:filehash(HERE/f) for f in ('e28_depth.py','protocol_e28.json','e27_reasoning.py','reasoning_processor.py','clrs_reference.py')},torch=torch.__version__,device=torch.cuda.get_device_name())
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');results=[];data_cache={};data_hashes={}
    for item in source['results']:
        seed=item['seed']
        if seed not in data_cache:
            path=args.source/f'{seed}-data.pt';data_hashes[seed]=filehash(path);data_cache[seed]=torch.load(path,weights_only=False);assert data_cache[seed]['identity']==item['source']
        results.append(run_source(item,data_cache[seed],cfg,scfg,args.out,args.preflight))
    assert filehash(source_path)==source_hash and all(filehash(args.source/f'{seed}-data.pt')==h for seed,h in data_hashes.items());complete=dict(manifest=manifest,source_data_hashes=data_hashes,results=results,status='PREFLIGHT_ONLY' if args.preflight else 'E28_COMPLETE');(args.out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n');print(json.dumps(dict(event=complete['status'],sources=len(results),audit='passed')),flush=True)
if __name__=='__main__':main()
