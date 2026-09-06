from pathlib import Path
import argparse,json,shutil
import numpy as np,torch
from e28_depth import state_hash
from e27_reasoning import filehash,digest
HERE=Path(__file__).parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);args=p.parse_args();torch.set_num_threads(1);full=json.loads((args.run/'complete.json').read_text());assert full['status']=='E28_COMPLETE';cfg=full['manifest']['config'];results=full['results'];assert len(results)==12
    for f,h in full['manifest']['code_identity'].items():assert filehash(HERE/f)==h
    source_path=Path(full['manifest']['source_path']);assert filehash(source_path)==full['manifest']['source_complete_sha256'];source=json.loads(source_path.read_text());sources={(x['seed'],x['arm']):x for x in source['results']};data={};audit=[]
    for seed,h in full['source_data_hashes'].items():
        p=source_path.parent/f'{seed}-data.pt';assert filehash(p)==h;data[int(seed)]=torch.load(p,weights_only=False)
    for result in results:
        cp=Path(result['source_checkpoint']);assert filehash(cp)==result['source_checkpoint_sha256'];saved=torch.load(cp,map_location='cpu',weights_only=False);assert state_hash(saved)==result['loaded_state_sha256'];src=sources[(result['seed'],result['source_arm'])];assert result['source_checkpoint_sha256']==src['stages'][-1]['checkpoint_sha256'];assert result['weights_optimizer_memory_rng_unchanged'];messages=0;decoders=0;queries=0;seconds=0
        for key,rows in result['grid'].items():
            d=data[result['seed']]['evaluation'][key];n=d['w'].shape[1];assert set(rows)==set(map(str,cfg['fixed_depths']))
            for depth,row in rows.items():
                depth=int(depth);pred=np.array(row['predictions']);equal=pred==d['y'];assert row['parent_accuracy']==float(equal.mean()) and row['canonical_graph_accuracy']==float(equal.all(1).mean());assert row['steps']==depth;assert row['message_candidates']==len(pred)*n*n*depth and row['decoder_candidates']==len(pred)*n*n;assert row['functional_graph_accuracy']<=1-row['invalid_tree_fraction']+1e-12;messages+=row['message_candidates'];decoders+=row['decoder_candidates'];queries+=row['graphs'];seconds+=row['neural_query_seconds']
            original=2 if result['source_arm'].startswith('short') else n;assert rows[str(original)]['predictions']==src['stages'][-1]['evaluation'][key]['predictions']
        assert messages==result['executed_message_candidates'] and decoders==result['executed_decoder_candidates'] and queries==result['executed_graph_queries'];assert abs(seconds-result['executed_query_seconds'])<1e-7
        for policy,cells in result['policies'].items():
            for key,row in cells.items():assert all(v==result['grid'][key][str(row['depth'])][k] for k,v in row.items() if k!='depth')
        audit.append(dict(seed=result['seed'],source_arm=result['source_arm'],state_hash_verified=True,all_original_predictions_exact=True,grid_cells=len(result['grid'])*len(cfg['fixed_depths']),unique_queries=queries,work_counts_exact=True))
    summary={};metrics=('functional_graph_accuracy','parent_accuracy','invalid_tree_fraction','neural_query_seconds','message_candidates','decoder_candidates')
    for arm in cfg['source_arms']:
        group=[x for x in results if x['source_arm']==arm];policies={};grid={}
        for policy in cfg['primary_policies']:
            policies[policy]={key:{m:float(np.mean([x['policies'][policy][key][m] for x in group])) for m in metrics} for key in group[0]['grid']}
        for key in group[0]['grid']:
            grid[key]={str(depth):{m:float(np.mean([x['grid'][key][str(depth)][m] for x in group])) for m in metrics} for depth in cfg['fixed_depths']}
        summary[arm]=dict(policies=policies,grid=grid,diagnostic_query_seconds=float(np.mean([x['executed_query_seconds'] for x in group])),diagnostic_message_candidates=int(group[0]['executed_message_candidates']))
    contrasts=[]
    for x in results:
        for key in x['grid']:
            r=x['policies'];n=int(key.split('-')[1]);contrasts.append(dict(seed=x['seed'],source_arm=x['source_arm'],cell=key,public_2N_minus_N_functional=r['public_2N'][key]['functional_graph_accuracy']-r['public_N'][key]['functional_graph_accuracy'],training_depth_minus_N_functional=r['training_depth'][key]['functional_graph_accuracy']-r['public_N'][key]['functional_graph_accuracy'],fixed16_minus_N_parent=r['fixed16'][key]['parent_accuracy']-r['public_N'][key]['parent_accuracy']))
    out=HERE/'results';shutil.copy2(args.run/'complete.json',out/'e28_complete.json');(out/'e28_audit.json').write_text(json.dumps(audit,indent=2)+'\n');(out/'e28_summary.json').write_text(json.dumps(dict(arms=summary,contrasts=contrasts),indent=2)+'\n')
    lines=['# E28 — frozen recurrence-depth attribution','',f'Source frozen at `{full["manifest"]["git_commit"]}`. All twelve final E27 checkpoints were evaluated at depths 2, 8, 16, 32, 64 and 128; 864 source/size/task/family/depth cells and 20,736 graph queries. No parameters, optimizer moments, episodic memory or RNG changed. The broad architecture goal remains active.','',
    'These are the same saved E27 queries, with depth intervened on after training. Every original-budget prediction reproduces E27 exactly. This is causal attribution of inference depth on those states, not an untouched new benchmark or evidence of newly learned competence. No depth is selected using a correct answer.','',
    '## Deployable budget comparisons','',
    'Entries are mean whole-graph correctness across three seeds, each with 24 graphs per cell. Policy costs use their actual grid rows; aliases reuse work rather than counting another execution. Training depth is two for short models and sixteen for size-trained models.','',
    '| Source | Budget | Dijkstra N16 chain | Dijkstra N64 random | Dijkstra N64 chain | Prim N16 random | Prim N64 random | Query seconds / full 288-graph set |','|---|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():
        for policy,cells in s['policies'].items():
            keys=['0-16-chain','0-64-random','0-64-chain','1-16-random','1-64-random'];vals=[100*cells[k]['functional_graph_accuracy'] for k in keys];seconds=sum(v['neural_query_seconds'] for v in cells.values());lines.append('| '+arm+' | '+policy+' | '+' | '.join(f'{v:.1f}%' for v in vals)+f' | {seconds:.3f} |')
    lines+=['','## Does additional recurrence preserve a familiar-length solution?','','Dijkstra N16 chain correctness, without changing graph, weights, memory or decoder:','','| Source | 2 steps | 8 steps | 16 steps | 32 steps | 64 steps | 128 steps |','|---|---:|---:|---:|---:|---:|---:|']
    for arm,s in summary.items():lines.append('| '+arm+' | '+' | '.join(f'{100*s["grid"]["0-16-chain"][str(d)]["functional_graph_accuracy"]:.1f}%' for d in cfg['fixed_depths'])+' |')
    positive=sum(c['public_2N_minus_N_functional']>0 for c in contrasts);negative=sum(c['public_2N_minus_N_functional']<0 for c in contrasts)
    lines+=['',f'Doubling public-N depth improves functional graph accuracy in {positive}/{len(contrasts)} source/task/size/family cells and worsens it in {negative}/{len(contrasts)}; the rest tie. These cells share models and graphs and are not independent replications. Raw per-seed contrasts are preserved.','',
    'Short fixed-depth local processing still has a structural communication limit on sufficiently long chains. Conversely, increasing a shared transition’s execution length does not guarantee that it preserves an answer it already learned to produce. These interventions cannot fully distinguish optimization, representation, state-transition instability, and training-depth specialization. They do not establish that randomized-depth training or another normalization would repair transfer.','',
    '## Cost and state audit','',
    f'The full diagnostic executed {sum(x["executed_graph_queries"] for x in results):,} graph queries, {sum(x["executed_message_candidates"] for x in results):,} dense forward message candidates and {sum(x["executed_decoder_candidates"] for x in results):,} decoder candidates. Recorded neural query time totals {sum(x["executed_query_seconds"] for x in results):.2f} seconds. This includes every tested depth; the per-policy table is not the cost of running the whole diagnostic. Message candidates are not measured FLOPs. GPU timing is hardware-specific and includes transfer of predicted pointers; reference validation and source setup are separate.','',
    'All twelve checkpoint file hashes and loaded model/optimizer/memory/RNG state hashes remain equal to their source. All source datasets and completion records retain their hashes. Every original E27 query prediction is exact, all grid work counts and recorded parent metrics check, and primary policy rows match their fixed-depth executions. No retraining, checkpoint selection or test-dependent halting was performed.','']
    (HERE/'E28-RESULT.md').write_text('\n'.join(lines))
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.3));colors=['#64748b','#f59e0b','#2563eb','#059669']
    for arm,c in zip(cfg['source_arms'],colors):
        s=summary[arm]
        for ax,key,metric in [(axes[0],'0-16-chain','functional_graph_accuracy'),(axes[1],'0-64-chain','functional_graph_accuracy'),(axes[2],'0-64-random','parent_accuracy')]:ax.plot(cfg['fixed_depths'],[100*s['grid'][key][str(d)][metric] for d in cfg['fixed_depths']],'o-',label=arm,color=c);ax.set_xscale('log',base=2);ax.set_xticks(cfg['fixed_depths'],list(map(str,cfg['fixed_depths'])));ax.set_ylim(-3,103);ax.set_xlabel('Recurrent steps')
    axes[0].set(title='N16 chains: complete correct solution',ylabel='Accuracy (%)');axes[1].set_title('N64 chains: complete correct solution');axes[2].set_title('N64 random: individual parent accuracy');axes[0].legend(fontsize=8);fig.suptitle('E28: fixed weights, changed inference depth',fontsize=13);fig.tight_layout();fig.savefig(out/'e28_depth.png',dpi=170);plt.close(fig)
    print(json.dumps(dict(status='E28_REPORT_AUDIT_PASSED',source_checks=len(audit),doubling_improves_cells=positive,doubling_worsens_cells=negative,total_cells=len(contrasts)),indent=2))
if __name__=='__main__':main()
