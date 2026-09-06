from pathlib import Path
import json,hashlib,shutil,statistics as st
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e25_query_routing import MemoryRouter,queries,run_queries,model_hash
HERE=Path(__file__).resolve().parent;RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e25')
torch.set_num_threads(1)
d=json.loads((RUNS/'complete.json').read_text());rs=d['records'];spec=d['protocol'];assert len(rs)==64 and not any(r['failed'] for r in rs)
for n,h in d['identity'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
checks=[]
for source in d['sources']:
    path=Path(source['checkpoint']);assert hashlib.sha256(path.read_bytes()).hexdigest()==source['checkpoint_sha256'];l=torch.load(path,weights_only=False)['learner'];models=list(l.models)
    if l.fast_model is not None:models.append(l.fast_model)
    active=l.fast_model if l.fast_model is not None else l.models[l.active];initial=model_hash(models)
    for arm in spec['arms']:
        cases=[r for r in rs if r['seed']==source['seed'] and r['source_arm']==source['source_arm'] and r['arm']==arm]
        p=Path(cases[0]['router_checkpoint']);packed=torch.load(p,weights_only=False);assert packed['identity']==d['identity'];router=packed['router']
        if router:assert router.tensor_bytes()==cases[0]['extra_routing_tensor_bytes'] and router.fit_cost==cases[0]['fit_cost']
        for r in cases:
            q=queries(spec,r['seed'],r['query_law']);result=run_queries(router,active,models,q)
            assert result['per_query_errors']==r['per_query_errors'] and result['query_cost']==r['query_cost'] and len(r['per_query_errors'])==4096,(r['seed'],r['source_arm'],r['arm'],r['query_law'],max(abs(a-b) for a,b in zip(result['per_query_errors'],r['per_query_errors'])))
            if source['experts']==1:
                base=next(b for b in rs if b['seed']==r['seed'] and b['source_arm']==r['source_arm'] and b['arm']=='active' and b['query_law']==r['query_law']);assert base['per_query_errors']==r['per_query_errors']
        assert model_hash(models)==initial
        checks.append(dict(seed=source['seed'],source_arm=source['source_arm'],arm=arm,router_checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),both_query_laws_exact_after_restore=True))
summary=[]
for source_arm in spec['source_arms']:
    for law in spec['query_laws']:
        for arm in spec['arms']:
            cases=[r for r in rs if r['source_arm']==source_arm and r['query_law']==law and r['arm']==arm]
            summary.append(dict(source_arm=source_arm,law=law,arm=arm,mse=st.mean(r['mse'] for r in cases),rare_mse=st.mean(r['rare_mse'] for r in cases),common_mse=st.mean(r['common_mse'] for r in cases),
                query_seconds=st.mean(r['query_seconds'] for r in cases),fit_seconds=st.mean(r['fit_cost'].get('fit_seconds',0)+r['fit_cost'].get('target_seconds',0) for r in cases),
                extra_tensor_bytes=st.mean(r['extra_routing_tensor_bytes'] for r in cases),query_expert_examples=st.mean(r['query_cost']['expert_examples'] for r in cases),
                query_router_examples=st.mean(r['query_cost']['router_examples'] for r in cases),query_distance_pairs=st.mean(r['query_cost']['distance_pairs'] for r in cases)))
look={(r['source_arm'],r['law'],r['arm']):r for r in summary}
report=['# E25: observed-memory routing recovers access but does not restore lost models','',
'All 64 query cases completed. Eight frozen E24 memory states are paired across four selectors and two fresh query laws. Stored experts, optimizers and experiences are unchanged. Fitting uses only observed reservoir inputs and noisy targets; final query outcomes, true region flags and E24 probe labels do not train or select a router. Three merged states have a single expert and explicitly bypass all routing work. The final non-merging state for seed 24104 also retains a temporary expert; it is included and charged.', '',
'| Source memory | Query law | Selector | Mean MSE | Rare MSE | Common MSE |', '|---|---|---|---:|---:|---:|']
for r in summary:report.append(f"| {r['source_arm']} | {r['law']} | {r['arm']} | {r['mse']:.6f} | {r['rare_mse']:.6f} | {r['common_mse']:.6f} |")
report+=['','## Access repair and its counterexamples','']
for source in spec['source_arms']:
    for law in spec['query_laws']:
        b=look[source,law,'active'];c=look[source,law,'learned_top1'];near=look[source,law,'nearest_anchor']
        wins=sum(next(r for r in rs if r['seed']==s and r['source_arm']==source and r['query_law']==law and r['arm']=='learned_top1')['mse']<next(r for r in rs if r['seed']==s and r['source_arm']==source and r['query_law']==law and r['arm']=='active')['mse'] for s in spec['source_seeds'])
        report.append(f"- {source}, {law}: learned top-one changes mean error {c['mse']/b['mse']-1:+.1%} versus active ({wins}/4 strict wins) and {c['mse']/near['mse']-1:+.1%} versus nearest-anchor.")
report+=['',
'Non-merging memory benefits from input-dependent access on every world and both supports, without changing any expert. The simpler nearest-anchor selector recovers much of the gain, so neural fitting is not the sole explanation. Top-one routing improves aggregate MSE versus nearest-anchor, but nearest-anchor wins individual comparisons, including shifted seed 24101 and both supports of the only nontrivial merged source. The learned mixture does not uniformly improve on top-one selection.', '',
'The three one-expert merged states produce exactly the same predictions under every selector. Their missing accuracy cannot be recovered by selecting a different retained model. The remaining merged state has much lower whole-query error with routing but higher rare-query error on original support: top-one increases rare MSE from 0.049869 to 0.060950. Whole-query improvement therefore does not establish rare-query preservation.', '',
'Non-merging seed 24102 retains high rare error under all selectors. Access does not restore competence that is absent from the available experts. All arms also face higher errors after support shift; new query frequencies and shifted coordinates do not create newly learned expertise. Eight frozen source states and correlated query summaries do not establish broad transfer or statistical significance.', '',
'## Complete costs distinguish a learned router from cheap access','',
'| Source | Selector | Fit + target construction seconds | Query seconds / 4096, original support | Additional persistent tensor bytes |', '|---|---|---:|---:|---:|']
for source in spec['source_arms']:
    for arm in spec['arms']:
        r=look[source,'original_support',arm];report.append(f"| {source} | {arm} | {r['fit_seconds']:.6f} | {r['query_seconds']:.6f} | {r['extra_tensor_bytes']:,.0f} |")
non=look['local_replay_isolation','original_support','learned_top1'];base=look['local_replay_isolation','original_support','active'];near=look['local_replay_isolation','original_support','nearest_anchor']
top_full=non['fit_seconds']+sum(look['local_replay_isolation',law,'learned_top1']['query_seconds'] for law in spec['query_laws']);near_full=near['fit_seconds']+sum(look['local_replay_isolation',law,'nearest_anchor']['query_seconds'] for law in spec['query_laws'])
report+=['',f'For non-merging states, top-one query time is {non["query_seconds"]/base["query_seconds"]:.2f} times active selection despite the same selected-expert example count. It uses less routing-cache memory and query time than nearest-anchor on these means, but fitting plus both 4,096-query laws costs {top_full/near_full:.2f} times nearest-anchor execution. This finite assay does not repay the neural fitting cost relative to the simpler selector. The listed extra bytes are added to the full inherited E24 state; they are not total model memory.', '',
'Target construction evaluates all K experts on every retained observation. Nontrivial neural routers add 256 backward/optimizer steps and 16,384 router training examples. Top-one evaluates one selected expert per query; the mixture evaluates every expert. Nearest-anchor computes every query-to-memory distance pair. All counts, transient training-prediction cache bytes, router/Adam state, source-loading time and inherited E24 work are retained. Fitting is shared across the two query laws and must be charged once per source/selector, not twice and not zero times. Privileged diagnostic evaluations and their timings are separate from operational queries.', '',
'Full process memory, peak autograd workspace, complete FLOPs, energy and eventual deployment amortization are unmeasured. Millisecond CPU query loops provide local comparisons, not production latency guarantees. The original acquisition and E24 memory costs are inherited, not erased by starting this experiment from checkpoints.', '',
'## Audit and next requirement','',
'With the frozen single-thread CPU setting, all 32 saved selector states reproduce both complete 4,096-query error sequences and operation counts exactly after restoration. An initial audit using the default thread setting failed exact floating equality; restoring the experiment\'s single-thread setting resolves it without changing any scored source or result. Source checkpoint hashes, unchanged expert tensors and source counters are checked. One-expert bypass predictions are exactly equal to active selection. Preflight exercises actual learned parameter changes, nearest-anchor recovery on stored dummy inputs, exact restored predictions and a double-precision finite-difference mixture gradient. The two fresh query laws never supply training labels.', '',
'This closes a scoped causal access gap. It does not demonstrate continual routing under changing experts, repair information loss, learn a new world, improve its own learning algorithm or establish ordered general reasoning. Mixture-of-experts and nearest-neighbor routing are established mechanisms, not novelty claims. The next integration must test this access mechanism as models and memory change, against the cheaper selector, and then move beyond supervised retrieval to the missing ordered reasoning and closed-loop components.']
for n in ('complete','preflight'):shutil.copyfile(RUNS/f'{n}.json',HERE/f'results/e25_{n}.json')
(HERE/'results/e25_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(HERE/'results/e25_audit.json').write_text(json.dumps(dict(identity=d['identity'],checks=checks),indent=2)+'\n');(HERE/'E25-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors=['#777777','#009E73','#0072B2','#D55E00']
for i,arm in enumerate(spec['arms']):
    vals=[look[src,law,arm]['mse'] for src in spec['source_arms'] for law in spec['query_laws']]
    axes[0].bar(np.arange(4)+(i-1.5)*.19,vals,.19,label=arm,color=colors[i])
    r=look['local_replay_isolation','original_support',arm];axes[1].bar(i,r['fit_seconds'],color='#CC79A7');axes[1].bar(i,sum(look['local_replay_isolation',law,arm]['query_seconds'] for law in spec['query_laws']),bottom=r['fit_seconds'],color=colors[i])
axes[0].set_xticks(range(4),['No merge / original','No merge / shifted','Merge / original','Merge / shifted'],rotation=15);axes[0].set_ylabel('Mean fresh-query MSE');axes[0].legend(fontsize=8);axes[0].set_title('Retained experts can support better selection')
axes[1].set_xticks(range(4),spec['arms'],rotation=15);axes[1].set_ylabel('Seconds: fit once + both query laws');axes[1].set_title('Neural fit (pink) has not amortized vs nearest')
fig.suptitle('E25 • eight frozen memories • 64 complete query cases • no expert learning');fig.tight_layout();fig.savefig(HERE/'results/e25_query_routing.png',dpi=160)
print('\n'.join(report[23:36]));print('AUDIT_OK',len(checks),'cases',len(rs),'fit_plus_query_ratio_to_nearest',top_full/near_full)
