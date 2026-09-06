"""E29 full-state audit and report; all sources, stages and inference policies."""
from pathlib import Path
import argparse,json,shutil
import numpy as np,torch
from e27_reasoning import Reservoir,digest,filehash,tensorbytes
from e28_depth import state_hash
from e29_variable_depth import depth_schedule
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1);full=json.loads((args.run/'complete.json').read_text());assert full['status']=='E29_COMPLETE';cfg=full['manifest']['config'];results=full['results'];assert len(results)==9 and len(full['audit'])==9 and len(full['pair_checks'])==9
    for f,h in full['manifest']['code_identity'].items():assert filehash(HERE/f)==h
    checks=[];pairs=[];b=cfg['current_batch'];updates=cfg['updates_per_stage'];n=cfg['train_nodes'];all_stages=len(cfg['stages'])
    for seed in cfg['seeds']:
        d=torch.load(args.run/f'{seed}-data.pt',weights_only=False);train=d['train'];ev=d['evaluation'];ident=d['identity'];assert ident['train_sha256']==digest([train[k] for k in ('w','s','t','y')]);assert all(ident['eval_sha256'][key]==digest([cell[k] for k in ('w','s','t','y')]) for key,cell in ev.items());group=[x for x in results if x['seed']==seed];assert len({x['initial_model_sha256'] for x in group})==1
        memory=Reservoir(cfg['replay_capacity_graphs'],n,seed+400000);memories={}
        for step in range(all_stages*updates):
            memory.sample(cfg['replay_batch']);memory.add({k:train[k][step*b:(step+1)*b] for k in ('w','s','t','y')})
            if (step+1)%updates==0:memories[(step+1)//updates-1]=memory.state()
        source_states={}
        for x in group:
            arm=x['arm'];assert x['source']==ident and x['parameters']==15177;schedule=x['depth_schedule'];assert schedule==depth_schedule(seed,arm,cfg);assert len(schedule)==all_stages*updates and x['recurrence_steps_total']==sum(schedule)
            if arm=='variable_replay':
                for stage in range(all_stages):
                    arr=schedule[stage*updates:(stage+1)*updates];assert arr[0]==24 and arr[-1]==24 and min(arr)>=16 and max(arr)<=32 and sum(arr)==24*updates
            elif arm=='fixed24_replay':assert set(schedule)=={24}
            else:assert arm=='fixed16_replay' and set(schedule)=={16}
            for stage in x['stages'][1:]:
                index=stage['stage'];cp=Path(stage['checkpoint']);assert filehash(cp)==stage['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);num=(index+1)*updates;current=num*b;replay=(num-1)*cfg['replay_batch'];presentations=current+replay;expected_messages=sum((b+(cfg['replay_batch'] if i else 0))*n*n*depth for i,depth in enumerate(schedule[:num]));assert saved['depth_schedule']==schedule
                assert saved['position']==current and saved['trained_graph_presentations']==presentations and saved['replay_graph_presentations']==replay and saved['message_candidates']==expected_messages and saved['decoder_candidates']==presentations*n*n
                assert all(int(v['step'])==num for v in saved['optimizer']['state'].values());assert all(torch.isfinite(v).all() for v in saved['model'].values());assert state_hash(saved['memory'])==state_hash(memories[index]);source_states[(arm,index)]=saved
                for policy,cells in stage['evaluation'].items():
                    assert policy in cfg['evaluation_policies']
                    for key,cell in cells.items():
                        pred=np.array(cell['predictions']);equal=pred==ev[key]['y'];nodes=pred.shape[1];depth=nodes if policy=='public_N' else 24 if policy=='fixed24' else 2*nodes
                        assert cell['steps']==depth and cell['message_candidates']==len(pred)*nodes*nodes*depth and cell['decoder_candidates']==len(pred)*nodes*nodes
                        assert cell['parent_accuracy']==float(equal.mean()) and cell['canonical_graph_accuracy']==float(equal.all(1).mean());assert cell['functional_graph_accuracy']>=cell['canonical_graph_accuracy']-1e-12 and cell['functional_graph_accuracy']<=1-cell['invalid_tree_fraction']+1e-12
                checks.append(dict(seed=seed,arm=arm,stage=index,checkpoint_sha256=stage['checkpoint_sha256'],optimizer_steps=num,messages=expected_messages,memory_reconstructed_exact=True))
            expected_bytes=tensorbytes(saved['model'])+tensorbytes(saved['optimizer'])+tensorbytes(saved['memory'])+tensorbytes(saved['torch_rng'])+tensorbytes(saved['cuda_rng']);assert x['persistent_tensor_bytes']==expected_bytes
        for stage in range(all_stages):
            fixed=source_states[('fixed24_replay',stage)];variable=source_states[('variable_replay',stage)];assert fixed['message_candidates']==variable['message_candidates'];assert all(state_hash(source_states[(arm,stage)]['memory'])==state_hash(fixed['memory']) for arm in cfg['arms']);pairs.append(dict(seed=seed,stage=stage,all_replay_states_exact=True,variable_fixed24_message_counts_exact=True,message_candidates=fixed['message_candidates']))
    summary={};metrics=('parent_accuracy','functional_graph_accuracy','canonical_graph_accuracy','invalid_tree_fraction','neural_query_seconds','message_candidates')
    for arm in cfg['arms']:
        group=[x for x in results if x['arm']==arm];stages={}
        for stage in (-1,0,1,2):
            stages[str(stage)]={policy:{key:{m:float(np.mean([x['stages'][stage+1]['evaluation'][policy][key][m] for x in group])) for m in metrics} for key in group[0]['stages'][stage+1]['evaluation'][policy]} for policy in cfg['evaluation_policies']}
        trials=[]
        for x in group:
            e=[s['evaluation']['public_N'] for s in x['stages'][1:]];d0=e[0]['0-16-random']['functional_graph_accuracy'];m1=e[1]['1-16-random']['functional_graph_accuracy'];d1=e[1]['0-16-random']['functional_graph_accuracy'];m2=e[2]['1-16-random']['functional_graph_accuracy'];trials.append(dict(seed=x['seed'],dijkstra_acquired=d0>=.8,mst_acquired=m1>=.8,dijkstra_drop_after_switch=d0-d1,mst_drop_after_return=m1-m2,dijkstra_retention_pass_given_acquisition=(d0-d1<=.05) if d0>=.8 else None,mst_retention_pass_given_acquisition=(m1-m2<=.05) if m1>=.8 else None,final_n64_random_both_pass=all(e[2][f'{t}-64-random']['functional_graph_accuracy']>=.8 for t in (0,1)),final_n64_chain_both_pass=all(e[2][f'{t}-64-chain']['functional_graph_accuracy']>=.8 for t in (0,1))))
        summary[arm]=dict(stages=stages,trials=trials,**{m:float(np.mean([x[m] for x in group])) for m in ('training_seconds','trained_graph_presentations','current_graph_presentations','replay_graph_presentations','optimizer_steps','message_candidates','decoder_candidates','persistent_tensor_bytes','cuda_peak_training_allocated_bytes')},total_evaluation_query_seconds=float(np.mean([sum(cell['neural_query_seconds'] for s in x['stages'] for cells in s['evaluation'].values() for cell in cells.values()) for x in group])))
    contrasts=[]
    for seed in cfg['seeds']:
        fixed=next(x for x in results if x['seed']==seed and x['arm']=='fixed24_replay');var=next(x for x in results if x['seed']==seed and x['arm']=='variable_replay')
        for policy in cfg['evaluation_policies']:
            for key in fixed['stages'][-1]['evaluation'][policy]:
                a=fixed['stages'][-1]['evaluation'][policy][key];v=var['stages'][-1]['evaluation'][policy][key];contrasts.append(dict(seed=seed,policy=policy,cell=key,variable_minus_fixed24_functional=v['functional_graph_accuracy']-a['functional_graph_accuracy'],variable_minus_fixed24_parent=v['parent_accuracy']-a['parent_accuracy']))
    out=HERE/'results';shutil.copy2(args.run/'complete.json',out/'e29_complete.json');(out/'e29_audit.json').write_text(json.dumps(dict(checkpoints=checks,paired_stages=pairs,final_restore=full['audit']),indent=2)+'\n');(out/'e29_summary.json').write_text(json.dumps(dict(arms=summary,contrasts=contrasts),indent=2)+'\n')
    lines=['# E29 — variable-depth continual graph training','',f'Source frozen at `{full["manifest"]["git_commit"]}`. All nine training cases and 27 stage checkpoints completed. Three fresh seeds compare fixed16, fixed24 and variable16–32 recurrence with identical current observations, replay, initialization, model and optimizer. The full goal remains active.','',
    'The variable schedule pairs r with 48-r and fixes first/last update at24, yielding exactly the fixed24 graph-weighted message count at every stage, including the first update without replay. This matches recurrence message work and corresponding backpropagation, not measured FLOPs or wall time. All arms retain weights, optimizer and a 128-graph reservoir across Dijkstra → Prim → Dijkstra. Each stage has 512 updates with eight new graphs and up to eight replay graphs per update. Final parent targets provide supervision; there are no intermediate hints, target-derived iteration lengths or test-selected checkpoints.','',
    'This is a bounded use of randomized-depth training described in the supplied TTC-LR work. It is not a Raven reproduction, a novel training procedure or recursive self-improvement. Cells average three seed fractions from 24 graphs per seed, so related cells must not be treated as independent replications.','',
    'Variable depth gives a scoped repair of inference-depth robustness, not the full reasoning claim. At 32 inference steps, final N16 Dijkstra chain correctness is 95.8% versus 63.9% for equal-work fixed24; the paired gain occurs in two seeds, with a 4.2-point loss in the first. Under public-N inference, final random-N16 Dijkstra correctness is 50.0% versus 29.2% for fixed24, but the cheaper fixed16 reaches 54.2%. Every final N64 task/family/policy cell has zero functionally correct graphs. No seed/arm meets the declared acquisition criteria.','',
    '## Primary public-N learning and retention','','| Arm | Dijkstra after first stage | Dijkstra after Prim | Prim after Prim | Dijkstra after return | Prim after return |','|---|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        get=lambda st,t:100*s['stages'][str(st)]['public_N'][f'{t}-16-random']['functional_graph_accuracy'];lines.append(f'| {arm} | {get(0,0):.1f}% | {get(1,0):.1f}% | {get(1,1):.1f}% | {get(2,0):.1f}% | {get(2,1):.1f}% |')
    lines+=['','## Final depth robustness and size transfer','','| Arm | Inference policy | Dijkstra N16 chain | Dijkstra N64 random | Dijkstra N64 chain | Prim N64 random | Prim N64 chain |','|---|---|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        for policy in cfg['evaluation_policies']:
            cells=s['stages']['2'][policy];keys=['0-16-chain','0-64-random','0-64-chain','1-64-random','1-64-chain'];lines.append('| '+arm+' | '+policy+' | '+' | '.join(f'{100*cells[k]["functional_graph_accuracy"]:.1f}%' for k in keys)+' |')
    lines+=['','All three policies were declared in advance. Fixed24 is an ordinary constant inference budget; public N and public 2N depend only on graph size. None chooses a depth from the correct answer. All policy executions, including overlapping depths on particular sizes, are charged. Stable answers at longer depth and learning a correct larger-graph algorithm are distinct outcomes.','',
    '## Charged work','','| Arm | Training seconds | All evaluation query seconds | Graph presentations | Forward message candidates | Persistent tensor bytes | Peak CUDA training bytes |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():lines.append(f'| {arm} | {s["training_seconds"]:.2f} | {s["total_evaluation_query_seconds"]:.2f} | {s["trained_graph_presentations"]:.0f} | {s["message_candidates"]:.0f} | {s["persistent_tensor_bytes"]:.0f} | {s["cuda_peak_training_allocated_bytes"]:.0f} |')
    lines+=['','Tensor bytes include model, optimizer, valid replay tensors and torch/CUDA RNG. They exclude Python objects, the externally generated depth-schedule list, runtime libraries and common source datasets; full process memory is not measured. CUDA peak allocation is measured separately. Graph generation and reference labels are common source costs recorded in raw identities; evaluation validation, checkpoint I/O and source setup are separate from these training/query timings. Message candidates are not measured FLOPs.','',
    '## Registered criteria','','Acquisition requires >=80% functional whole-graph correctness on random N16 inputs of the newly taught task. Retention permits <=5 percentage points lost only when acquisition occurred. N64 transfer requires >=80% functional correctness. Failure to acquire cannot be recast as a retention success.','']
    for arm,s in summary.items():
        t=s['trials'];lines.append(f'- {arm}: Dijkstra acquisition {sum(v["dijkstra_acquired"] for v in t)}/3; Prim acquisition {sum(v["mst_acquired"] for v in t)}/3; final both-task N64 random transfer {sum(v["final_n64_random_both_pass"] for v in t)}/3; N64 chain transfer {sum(v["final_n64_chain_both_pass"] for v in t)}/3.')
    lines+=['','## Audit','','All 27 stage checkpoint hashes, optimizer steps, finite weights, declared depth schedules and graph/message counters were checked. Replay contents and RNG were independently reconstructed from the original stream at each stage. Initialization and current data hashes match across arms; fixed24 and variable have exactly equal message counts in all nine paired stages. All nine restored final models reproduce all final predictions for all three policies exactly in the scored runner. Evaluation preserves model and torch/CUDA RNG state. The separate preflight additionally reproduces E27 baseline losses and model/optimizer/memory/RNG exactly.','']
    (HERE/'E29-RESULT.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4));colors=['#64748b','#2563eb','#059669']
    for arm,color in zip(cfg['arms'],colors):
        s=summary[arm];axes[0].plot([0,1,2],[100*s['stages'][str(st)]['public_N']['0-16-random']['functional_graph_accuracy'] for st in (0,1,2)],'o-',label=arm,color=color);axes[1].plot([0,1,2],[100*s['stages']['2'][policy]['0-16-chain']['functional_graph_accuracy'] for policy in cfg['evaluation_policies']],'o-',color=color)
    axes[0].set(title='Random N16 Dijkstra across tasks',ylabel='Functionally correct graphs (%)',xticks=[0,1,2],xticklabels=['Dijkstra','Prim','Dijkstra']);axes[0].legend(fontsize=8);axes[1].set(title='Final N16 chain depth robustness',xticks=[0,1,2],xticklabels=['N = 16','Fixed 24','2N = 32']);axes[0].set_ylim(-3,103);axes[1].set_ylim(-3,103)
    axes[2].bar(range(3),[summary[a]['training_seconds'] for a in cfg['arms']],color=colors);axes[2].set(title='Training time; fixed24/variable work matches',ylabel='Seconds',xticks=range(3),xticklabels=['Fixed16','Fixed24','Variable']);fig.suptitle('E29: varied training depth versus equal-work fixed recurrence',fontsize=13);fig.tight_layout();fig.savefig(out/'e29_variable_depth.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(status='E29_REPORT_AUDIT_PASSED',checkpoints=len(checks),paired_stages=len(pairs),criteria={arm:s['trials'] for arm,s in summary.items()}),indent=2))
if __name__=='__main__':main()
