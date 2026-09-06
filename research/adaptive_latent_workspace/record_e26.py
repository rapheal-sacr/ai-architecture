from pathlib import Path
import json,hashlib,statistics as st,shutil
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e26_online_access import OnlineAccess,core_state,equal,fresh_probes,probe
HERE=Path(__file__).resolve().parent;RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e26');torch.set_num_threads(1)
d=json.loads((RUNS/'complete.json').read_text());rs=d['records'];spec=d['protocol'];assert len(rs)==24 and not any(r['failed'] for r in rs)
for n,h in d['identity'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
checks=[];groups={};events=[]
for r in rs:
    path=Path(r['checkpoint']);p=torch.load(path,weights_only=False);l=p['learner'];router=p['router'];assert p['identity']==d['identity'] and l.updates==16384
    assert l.costs()==r['final_core_costs'] and router.counts==r['router_fit_cost'] and router.tensor_bytes()==r['router_tensor_bytes']
    result,_=probe(l,router,fresh_probes(spec,r['seed']));final=r['endpoints'][-1].copy();final.pop('continuation_batch');assert result==final
    key=(r['seed'],r['source_arm']);state=core_state(l)
    if key in groups:assert equal(state,groups[key][0]) and torch.equal(p['rng'],groups[key][1])
    else:
        groups[key]=(state,p['rng']);warm=torch.load(RUNS/f'{r["seed"]}-{r["source_arm"]}-reconstructed.pt',weights_only=False)
        events.append(dict(seed=r['seed'],source_arm=r['source_arm'],new_events=len(l.events)-len(warm['learner'].events),new_merges=l.merges-warm['learner'].merges,
            final_persistent_models=len(l.models),final_temporary=l.fast_model is not None))
    assert len(r['clean_batch_mse'])==8192 and all(torch.isfinite(t).all() for m in l.models for t in m.parameters())
    checks.append(dict(seed=r['seed'],source_arm=r['source_arm'],arm=r['arm'],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),final_probe_and_counters_exact=True,global_step=l.updates))
summary=[]
for source in spec['source_arms']:
    for arm in spec['arms']:
        cases=[r for r in rs if r['source_arm']==source and r['arm']==arm]
        summary.append(dict(source_arm=source,arm=arm,mse=st.mean(r['mean_clean_mse'] for r in cases),final_rare=st.mean(r['endpoints'][-1]['rare'] for r in cases),final_common=st.mean(r['endpoints'][-1]['common'] for r in cases),
            all_probe_rare=st.mean(e['rare'] for r in cases for e in r['endpoints']),seconds=st.mean(r['loop_seconds_including_audit']+r['router_bootstrap_seconds'] for r in cases),
            audit_seconds=st.mean(r['audit_seconds'] for r in cases),model_examples=st.mean(r['new_core_forward_examples']+r['query_cost']['expert_examples']+r['router_fit_cost']['target_expert_examples'] for r in cases),
            router_backward_batches=st.mean(r['router_fit_cost']['router_backward_batches'] for r in cases),router_train_examples=st.mean(r['router_fit_cost']['router_train_examples'] for r in cases),
            distance_pairs=st.mean(r['query_cost']['distance_pairs'] for r in cases),tensor_bytes=st.mean(r['final_core_costs']['stored_tensor_bytes']+r['router_tensor_bytes'] for r in cases)))
look={(r['source_arm'],r['arm']):r for r in summary}
report=['# E26: online routing preserves an access gain at substantial extra work','',
'All 24 continuations completed from eight matched E24 sources. The entire 8,192-batch prefix was regenerated exactly, every historical prediction error reproduced, and model/optimizer/anchor/control state matched each saved source before its missing global reservoir RNG was carried forward. Each continuation then learns for another 8,192 updates (262,144 observations), reaching 16,384 persistent updates. Four historical worlds are shared across source policies and access arms; these are not 24 independent new worlds.', '',
'Predictions use only preceding routing state and current inputs. After real outputs arrive, the unchanged E6 learner updates, then observed records can train the router. All three selectors finish with exactly equal underlying model, optimizer, memory, control and global-RNG states within each source. The access interface changes the predictions while preserving the underlying learning policy.', '',
'| Source memory | Access | Mean stream MSE | Mean final rare-query MSE | Mean final common-query MSE | Mean seconds incl. bootstrap/audit |', '|---|---|---:|---:|---:|---:|']
for r in summary:report.append(f"| {r['source_arm']} | {r['arm']} | {r['mse']:.7f} | {r['final_rare']:.6f} | {r['final_common']:.6f} | {r['seconds']:.3f} |")
report+=['','## What survives online use','']
for source in spec['source_arms']:
    b=look[source,'active'];n=look[source,'nearest_cached'];c=look[source,'online_top1']
    wins=sum(next(r for r in rs if r['seed']==s and r['source_arm']==source and r['arm']=='online_top1')['mean_clean_mse']<next(r for r in rs if r['seed']==s and r['source_arm']==source and r['arm']=='active')['mean_clean_mse'] for s in spec['source_seeds'])
    report.append(f"- {source}: online top-one changes mean stream error {c['mse']/b['mse']-1:+.1%} ({wins}/4 strict wins) and final rare error {c['final_rare']/b['final_rare']-1:+.1%} versus active. Runtime is {c['seconds']/b['seconds']:.2f} times and model examples {c['model_examples']/b['model_examples']:.2f} times active. Versus nearest-cache, stream error changes {c['mse']/n['mse']-1:+.1%} at {c['seconds']/n['seconds']:.2f} times runtime.")
report+=['',
'Non-merging memory retains a useful access benefit while its experts and reservoirs update. The learned router has lower stream error in all four such worlds than both alternatives. Refreshed nearest-anchor access lowers mean final rare error but raises mean stream error: a rare-retrieval gain is not automatically better performance under the actual observation frequencies. The learned router needs much more work and is not established as an efficient dominant solution.', '',
'Three merging sources remain one-expert cases throughout this continuation and produce exactly equal error sequences across access methods. Their online neural selector still retains a private RNG state even when no network is allocated; those bytes and execution overhead remain counted. In the remaining merging source, learned selection lowers stream error but slightly worsens final rare error versus active selection. Access still cannot restore information absent from every available expert.', '',
'Final rare competence remains poor in the second and fourth non-merging worlds despite routing. The continuation revisits the same fixed conditional functions and input means; it does not establish adaptation to new algorithms, new functions or a closed-loop environment. Region-balanced probes and the rare-event observation stream answer different questions. The best-region expert metric is a privileged single-expert diagnostic, not a bound on input-dependent combinations.', '',
'## Actual library changes','', '| Seed | Source | New events | New merges | Final persistent models | Live temporary model |', '|---:|---|---:|---:|---:|---|']
for e in events:report.append(f"| {e['seed']} | {e['source_arm']} | {e['new_events']} | {e['new_merges']} | {e['final_persistent_models']} | {e['final_temporary']} |")
report+=['',
'Event counts can include repeated checks rather than new knowledge. The actual changes above bound what this continuation tested. The dummy preflight additionally exercises temporary-expert and merge slot handling; passing that functional check is not broad evidence under repeated novel-task arrivals.', '',
'## All added work and retained state','',
'| Source | Access | New operational + fitting expert examples | Router training examples | Router backward batches | Query distance pairs | Final counted core + routing bytes |', '|---|---|---:|---:|---:|---:|---:|']
for r in summary:report.append(f"| {r['source_arm']} | {r['arm']} | {r['model_examples']:,.0f} | {r['router_train_examples']:,.0f} | {r['router_backward_batches']:,.0f} | {r['distance_pairs']:,.0f} | {r['tensor_bytes']:,.0f} |")
replay=sum(r['reconstruction_seconds'] for r in d['sources'])
report+=['',f'The eight required source reconstructions cost {replay:.2f} seconds of replay, plus separately recorded checkpoint loading and prefix generation. Reconstruction is shared once per source, not charged three times or omitted. Original E24 work is inherited separately. Router bootstrap is included in the main time table, and actual loop time includes audit probes. Probe expert/router/distance work and times are separately retained; the model-example table excludes those audit examples. New core inference counters exclude predictions because each access method counts its actual prediction work separately.', '',
'Target construction recomputes expert errors on the observed router training batches, including every neural bootstrap batch. Nothing is treated as a free oracle label. Final state includes the core optimizers/anchors and routing parameters, Adam state, normalization/cache tensors and private RNG where retained. Full process and peak autograd RAM, complete FLOPs, energy and production latency remain unmeasured. Small CPU runtimes and tensor counts do not establish hardware-independent efficiency.', '',
'## Verification and architecture boundary','',
'All 24 final checkpoints have 16,384 updates, finite persistent experts, matching source identity and exactly reproduced final probe values, core counters and router counts. All eight groups have exactly equal final underlying learning states and global RNG across selectors. The preflight verifies RNG isolation on each dummy step, valid slots, actual router training and exact saved/restored next-update behavior. Source reconstruction is a stronger check than assuming a serialized model contains its unstored RNG.', '',
'This integrates persistent online learned selection with real model updates and bounded replay. It is not recursive self-application, a new general learning algorithm, semantic compression or general reasoning. Gains coexist with cost and retention counterexamples. The next experiment should address the missing multi-step reasoning processor on externally specified problems, rather than continue tuning this supervised access assay.']
for n in ('complete','preflight'):shutil.copyfile(RUNS/f'{n}.json',HERE/f'results/e26_{n}.json')
(HERE/'results/e26_summary.json').write_text(json.dumps(dict(rows=summary,library_changes=events),indent=2)+'\n');(HERE/'results/e26_audit.json').write_text(json.dumps(dict(identity=d['identity'],checks=checks,identical_final_core_rng_groups=len(groups)),indent=2)+'\n');(HERE/'E26-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,3,figsize=(15,4.4));colors=['#777777','#009E73','#0072B2']
for i,arm in enumerate(spec['arms']):
    cases=[r for r in rs if r['source_arm']=='local_replay_isolation' and r['arm']==arm]
    axes[0].plot(np.arange(1,17)*512,np.mean([[e['rare'] for e in r['endpoints']] for r in cases],0),label=arm,color=colors[i])
    v=look['local_replay_isolation',arm];axes[1].bar(i,v['mse']/look['local_replay_isolation','active']['mse'],color=colors[i]);axes[2].bar(i,v['seconds']/look['local_replay_isolation','active']['seconds'],color=colors[i])
axes[0].set_xlabel('New online updates');axes[0].set_ylabel('Mean held-out rare-query MSE');axes[0].legend(fontsize=8);axes[0].set_title('Access gains persist while experts update')
for ax,title,ylabel in [(axes[1],'Rare access and stream error can disagree','Stream MSE / active'),(axes[2],'The learned router adds substantial work','Seconds / active')]:
    ax.set_xticks(range(3),spec['arms'],rotation=20);ax.set_title(title);ax.set_ylabel(ylabel);ax.axhline(1,color='black',lw=.7)
fig.suptitle('E26 • non-merging source means • exact source reconstruction and matched core states');fig.tight_layout();fig.savefig(HERE/'results/e26_online_access.png',dpi=160)
print('\n'.join(report[:24]));print('AUDIT_OK',len(checks),'groups',len(groups),'events',events)
