"""Audit and summarize all registered E27 cases, without selecting checkpoints."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np
import torch
from e27_reasoning import Reservoir,digest,filehash,tensorbytes
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1)
    complete=json.loads((args.run/'complete.json').read_text());assert complete['status']=='E27_COMPLETE';cfg=complete['manifest']['config'];results=complete['results'];assert len(results)==len(cfg['seeds'])*len(cfg['arms']);assert len(complete['audit'])==len(results)
    for f,h in complete['manifest']['code_identity'].items():assert filehash(HERE/f)==h
    checkpoints=[];pair_checks=[]
    for seed in cfg['seeds']:
        dataset=torch.load(args.run/f'{seed}-data.pt',map_location='cpu',weights_only=False);data=dataset['train'];ev=dataset['evaluation'];ident=dataset['identity'];assert ident['train_sha256']==digest([data[k] for k in ('w','s','t','y')]);assert all(ident['eval_sha256'][k]==digest([v[x] for x in ('w','s','t','y')]) for k,v in ev.items())
        group=[x for x in results if x['seed']==seed];assert len({x['initial_model_sha256'] for x in group})==1
        expected_memory=Reservoir(cfg['replay_capacity_graphs'],cfg['train_nodes'],seed+400000);memory_at_stage={};b=cfg['current_batch'];upd=cfg['updates_per_stage']
        for step in range(len(cfg['stages'])*upd):
            expected_memory.sample(cfg['replay_batch']);cur={k:data[k][step*b:(step+1)*b] for k in ('w','s','t','y')};expected_memory.add(cur)
            if (step+1)%upd==0:memory_at_stage[(step+1)//upd-1]=expected_memory.state()
        for result in group:
            assert result['parameters']==15177 and result['source']==ident
            steps=cfg['short_steps'] if result['arm'].startswith('short') else cfg['train_nodes']
            for stage in result['stages'][1:]:
                cp=Path(stage['checkpoint']);assert filehash(cp)==stage['checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);nupdates=(stage['stage']+1)*upd;current=nupdates*b;replay=(nupdates-1)*cfg['replay_batch'] if result['arm'].endswith('replay') else 0
                assert saved['position']==current and saved['trained_graph_presentations']==current+replay and saved['replay_graph_presentations']==replay
                assert saved['message_candidates']==(current+replay)*cfg['train_nodes']**2*steps and saved['decoder_candidates']==(current+replay)*cfg['train_nodes']**2
                assert all(int(v['step'])==nupdates for v in saved['optimizer']['state'].values());assert all(torch.isfinite(v).all() for v in saved['model'].values())
                if result['arm'].endswith('replay'):
                    expected=memory_at_stage[stage['stage']];actual=saved['memory'];assert all(torch.equal(actual[k],expected[k]) for k in ('w','s','t','y'));assert actual['seen']==expected['seen'] and actual['rng']==expected['rng']
                else:assert saved['memory']['size']==0
                for key,v in stage['evaluation'].items():
                    pred=np.array(v['predictions']);equal=pred==ev[key]['y'];assert v['parent_accuracy']==float(equal.mean());assert v['canonical_graph_accuracy']==float(equal.all(1).mean());assert v['functional_graph_accuracy']<=1-v['invalid_tree_fraction']+1e-12;assert v['functional_graph_accuracy']>=v['canonical_graph_accuracy']-1e-12
                checkpoints.append(dict(seed=seed,arm=result['arm'],stage=stage['stage'],hash=stage['checkpoint_sha256'],optimizer_steps=nupdates,counters_and_memory_exact=True))
            expected_bytes=tensorbytes(saved['model'])+tensorbytes(saved['optimizer'])+tensorbytes(saved['memory'])+tensorbytes(saved['torch_rng'])+tensorbytes(saved['cuda_rng']);assert result['persistent_tensor_bytes']==expected_bytes
        pair_checks.append(dict(seed=seed,identical_initial_model=True,all_current_data_hashes_exact=True,replay_contents_reconstructed_at_all_stages=True))
    summaries={}
    for arm in cfg['arms']:
        group=[x for x in results if x['arm']==arm];stages={}
        for stage in (-1,0,1,2):
            rows=[x['stages'][stage+1]['evaluation'] for x in group];stages[str(stage)]={key:{m:float(np.mean([r[key][m] for r in rows])) for m in ('parent_accuracy','canonical_graph_accuracy','functional_graph_accuracy','invalid_tree_fraction','neural_query_seconds','exact_reference_query_seconds')} for key in rows[0]}
        trials=[]
        for x in group:
            e=[v['evaluation'] for v in x['stages'][1:]];d0=e[0]['0-16-random']['functional_graph_accuracy'];m1=e[1]['1-16-random']['functional_graph_accuracy'];d1=e[1]['0-16-random']['functional_graph_accuracy'];m2=e[2]['1-16-random']['functional_graph_accuracy']
            trials.append(dict(seed=x['seed'],dijkstra_acquired=d0>=.8,mst_acquired=m1>=.8,dijkstra_drop_after_switch=d0-d1,mst_drop_after_return=m1-m2,dijkstra_retention_pass_given_acquisition=(d0-d1<=.05) if d0>=.8 else None,mst_retention_pass_given_acquisition=(m1-m2<=.05) if m1>=.8 else None,final_n64_random_both_pass=all(e[2][f'{t}-64-random']['functional_graph_accuracy']>=.8 for t in (0,1)),final_n64_chain_both_pass=all(e[2][f'{t}-64-chain']['functional_graph_accuracy']>=.8 for t in (0,1))))
        summaries[arm]=dict(stages=stages,trials=trials,**{m:float(np.mean([x[m] for x in group])) for m in ('training_seconds','trained_graph_presentations','current_graph_presentations','replay_graph_presentations','optimizer_steps','message_candidates','decoder_candidates','persistent_tensor_bytes','cuda_peak_training_allocated_bytes')},evaluation_seconds=float(np.mean([sum(v['neural_query_seconds'] for stage in x['stages'] for v in stage['evaluation'].values()) for x in group])))
    # Descriptive post-hoc breakdown of already-recorded predictions; no new queries.
    distance_bins=[(0,2),(3,4),(5,8),(9,16),(17,32),(33,1000)];distance_rows={}
    for result in results:
        dataset=torch.load(args.run/f'{result['seed']}-data.pt',map_location='cpu',weights_only=False)['evaluation']
        for key,cell in result['stages'][-1]['evaluation'].items():
            if not key.endswith('chain'):continue
            group_key=result['arm']+'-'+key
            row=distance_rows.setdefault(group_key,{f'{lo}-{hi}':dict(correct=0,nodes=0) for lo,hi in distance_bins})
            for w,source,target,pred in zip(dataset[key]['w'],dataset[key]['s'],dataset[key]['y'],cell['predictions']):
                distance=np.full(len(w),-1);distance[source]=0;queue=[int(source)]
                for v in queue:
                    for u in np.flatnonzero(w[v]):
                        if distance[u]<0:distance[u]=distance[v]+1;queue.append(int(u))
                assert np.all(distance>=0);correct=np.array(pred)==target
                for lo,hi in distance_bins:
                    mask=(distance>=lo)&(distance<=hi);row[f'{lo}-{hi}']['correct']+=int(correct[mask].sum());row[f'{lo}-{hi}']['nodes']+=int(mask.sum())
    for row in distance_rows.values():
        for v in row.values():v['accuracy']=v['correct']/v['nodes'] if v['nodes'] else None
    out=HERE/'results';out.mkdir(exist_ok=True);shutil.copy2(args.run/'complete.json',out/'e27_complete.json');(out/'e27_audit.json').write_text(json.dumps(dict(checkpoints=checkpoints,pair_checks=pair_checks,final_prediction_reproduction=complete['audit']),indent=2)+'\n');(out/'e27_summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    (out/'e27_distance_diagnostic.json').write_text(json.dumps(dict(status='DESCRIPTIVE_POST_HOC_ALREADY_RECORDED_PREDICTIONS',results=distance_rows),indent=2)+'\n')
    lines=['# E27 — final-target recurrent graph learning and retention','',f'Source frozen at `{complete["manifest"]["git_commit"]}`. All twelve cases and 36 stage checkpoints completed. The goal remains active. This is a small supervised graph pilot, not an integrated autonomous agent.','',
    'The same 15,177-parameter graph processor consumes Dijkstra → Prim → Dijkstra tasks, 512 updates per stage, eight new N16 graphs per update. Public-N recurrence uses 16 training steps versus two. Replay adds eight earlier graphs per update after the first, held in a 128-graph reservoir. All arms share current observations and initialization; replay trajectories are identical across recurrence depths. Explicit task IDs request algorithms; no gold hints or target-derived stopping lengths enter training or prediction.','',
    'Each reported cell averages three seed fractions from 24 fresh graphs per seed. These are small, correlated evaluations; no significance or broad generalization claim follows. No test-selected checkpoints or tuning were used.','',
    'The candidate fails the registered acquisition and N64 transfer criteria in every seed and arm. Longer recurrence plus replay reaches 100% final Dijkstra correctness on the tested N16 chains, but all arms score 0% functional correctness on every final N64 task/family cell. Thus familiar-length chain success does not establish a transferable algorithm. Longer recurrence takes about 3.6–3.7 times training time at eight times forward message candidates compared with its matched replay choice.','',
    '## Whole-graph correctness on random N16 graphs','',
    '| Arm | Dijkstra after first task | Dijkstra after MST | MST after MST | Dijkstra after return | MST after return |','|---|---:|---:|---:|---:|---:|']
    for arm,s in summaries.items():
        v=lambda st,t:s['stages'][str(st)][f'{t}-16-random']['functional_graph_accuracy']*100
        lines.append(f'| {arm} | {v(0,0):.1f}% | {v(1,0):.1f}% | {v(1,1):.1f}% | {v(2,0):.1f}% | {v(2,1):.1f}% |')
    lines+=['','## Larger inputs after the final stage','','| Arm | N64 random Dijkstra | N64 random MST | N64 chain Dijkstra | N64 chain MST |','|---|---:|---:|---:|---:|']
    for arm,s in summaries.items():
        vals=[s['stages']['2'][f'{t}-64-{family}']['functional_graph_accuracy']*100 for family in ('random','chain') for t in (0,1)];lines.append('| '+arm+' | '+' | '.join(f'{v:.1f}%' for v in vals)+' |')
    lines+=['','## Charged work','','| Arm | Train seconds | All evaluation seconds | Graph presentations | Message candidates | Persistent tensor bytes | Peak CUDA training bytes |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in summaries.items():lines.append(f'| {arm} | {s["training_seconds"]:.2f} | {s["evaluation_seconds"]:.2f} | {s["trained_graph_presentations"]:.0f} | {s["message_candidates"]:.0f} | {s["persistent_tensor_bytes"]:.0f} | {s["cuda_peak_training_allocated_bytes"]:.0f} |')
    lines+=['','Training message counts are dense forward candidates, with corresponding backpropagation; they are not measured FLOPs. Public-N recurrence uses eight times two-step message candidates during N16 training. A full loss trace, all evaluation/query counts and predictions are retained. Persistent tensor bytes include model, optimizer, used reservoir tensors and torch/CUDA RNG; they exclude Python object overhead and runtime libraries. Peak CUDA allocation is a separate training measure. Common dataset generation, label production, evaluation validation and checkpoint I/O are not hidden in these per-arm training timings; their available costs appear in source identities and raw records. All runtime measurements are hardware-specific.', '',
    'Exact CLRS source algorithms produce functionally correct targets, independently checked with NetworkX. Their CPU query timing includes the local probe adapter, excludes independent validation, and is stored separately per evaluation cell. This is not a speed comparison against an optimized or hardware-matched classical baseline. No published CLRS training result was reproduced.','',
    '## Registered criteria and falsification','','Acquisition requires at least 80% functional whole-graph correctness on random N16 tasks; retention allows at most five percentage points lost after switching; N64 transfer requires at least 80%. Retention is only scored as a pass/failure when the task was acquired. Earlier low accuracy followed by later low accuracy is not evidence of successful retention.','']
    for arm,s in summaries.items():
        trials=s['trials'];lines.append(f'- {arm}: Dijkstra acquired {sum(t["dijkstra_acquired"] for t in trials)}/3; MST acquired {sum(t["mst_acquired"] for t in trials)}/3; final both-task N64 random transfer {sum(t["final_n64_random_both_pass"] for t in trials)}/3; N64 chain transfer {sum(t["final_n64_chain_both_pass"] for t in trials)}/3.')
    lines+=['','The untrained two-step locality counterexample is structural. Longer recurrence removes that particular receptive-field limit, but a trained failure can still reflect optimization, representation, supervision or insufficient experience. This experiment does not distinguish all those causes. More recurrence alone is not a demonstrated general reasoning solution.','',
    '## Audit','','All 36 checkpoint hashes, optimizer step counts, graph/message counters, finite weights and recorded pointer metrics were verified. Reservoir contents and private RNG were reconstructed from the original stream at every stage; paired initialization and dataset hashes match. The scored runner restores all twelve final models and reproduces every saved final prediction exactly on the GPU. Evaluation checks preserve weights, memory sampling RNG and torch/CUDA RNG.','']
    (HERE/'E27-RESULT.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.4));colors=['#64748b','#f59e0b','#2563eb','#059669']
    for arm,color in zip(cfg['arms'],colors):
        s=summaries[arm];axes[0].plot([0,1,2],[100*s['stages'][str(st)]['0-16-random']['functional_graph_accuracy'] for st in (0,1,2)],'o-',label=arm,color=color);axes[1].plot([16,32,64],[100*s['stages']['2'][f'0-{n}-chain']['functional_graph_accuracy'] for n in (16,32,64)],'o-',label=arm,color=color)
    axes[0].set(title='Random N16: Dijkstra across tasks',ylabel='Functionally correct graphs (%)',xticks=[0,1,2],xticklabels=['Dijkstra','MST','Dijkstra']);axes[1].set(title='Final Dijkstra on long chains',xlabel='Nodes',xticks=[16,32,64]);axes[0].set_ylim(-3,103);axes[1].set_ylim(-3,103);axes[0].legend(fontsize=8)
    axes[2].bar(range(4),[summaries[a]['training_seconds'] for a in cfg['arms']],color=colors);axes[2].set(title='Training time (mean of 3 seeds)',ylabel='Seconds',xticks=range(4),xticklabels=['Short','Short\n+ replay','Size','Size\n+ replay'])
    fig.suptitle('E27: whole-graph correctness, transfer and charged training',fontsize=13);fig.tight_layout();fig.savefig(out/'e27_reasoning.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(status='E27_REPORT_AUDIT_PASSED',checkpoints=len(checkpoints),paired_seeds=len(pair_checks),criteria={a:s['trials'] for a,s in summaries.items()}),indent=2))
if __name__=='__main__':main()
