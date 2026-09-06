from pathlib import Path
import json,hashlib,shutil,statistics as st
import torch,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from e24_rare_merge_audit import generate,endpoint
from e6_sharing import GuardedSharing
HERE=Path(__file__).resolve().parent;RUNS=Path('/media/rapheal/New WD m2 Udemba Boys/AI Architecture Research/alw-runs/e24')
d=json.loads((RUNS/'complete.json').read_text());rs=d['records'];spec=d['protocol'];assert len(rs)==12 and not any(r['failed'] for r in rs)
for n,h in d['identity'].items():assert hashlib.sha256((HERE/n).read_bytes()).hexdigest()==h,n
checks=[]
for r in rs:
    path=Path(r['checkpoint']);packed=torch.load(path,weights_only=False);l=packed['learner'];assert packed['identity']==d['identity']
    assert l.updates==8192 and l.costs()==r['learner_costs'] and len(r['clean_batch_mse'])==8192
    assert all(torch.isfinite(p).all() for m in l.models for p in m.parameters())
    _,probes,_=generate(spec,r['seed'],0);ep=endpoint(l,probes);stored=r['endpoints'][-1].copy();stored.pop('batch');assert ep==stored
    anchors=[]
    if isinstance(l,GuardedSharing):
        assert len(l.models)<=2 and all(a.size<=128 for a in l.anchors)
        anchors=[dict(module=i,size=a.size,seen=a.count,rare_examples=int((a.x[:a.size,0]>0).sum())) for i,a in enumerate(l.anchors)]
        assert sum(a['event']['merged'] for a in r['audits'])==l.merges
    checks.append(dict(seed=r['seed'],arm=r['arm'],checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),restored_counters_and_probe_predictions_exact=True,anchor_coverage=anchors))
summary=[]
for arm in spec['arms']:
    cases=[r for r in rs if r['arm']==arm];a=[a for r in cases for a in r['audits']];accepted=[c for c in a if c['event']['merged']]
    summary.append(dict(arm=arm,mean_prequential_mse=st.mean(r['mean_clean_mse'] for r in cases),mean_rare_acquisition=st.mean(r['endpoints'][0]['operational']['rare'] for r in cases),
        mean_rare_final=st.mean(r['endpoints'][-1]['operational']['rare'] for r in cases),mean_common_final=st.mean(r['endpoints'][-1]['operational']['common'] for r in cases),
        final_rare_competent=sum(r['endpoints'][-1]['operational']['rare']<=spec['competence_mse'] for r in cases),
        checks=len(a),source_competent_checks=sum(c['source_rare_competent'] for c in a),accepted=len(accepted),accepted_source_competent=sum(c['source_rare_competent'] for c in accepted),
        material_proposal_losses=sum(c['material_rare_loss'] for c in a),accepted_material_losses=sum(c['accepted_material_rare_loss'] for c in a),
        mean_loop_seconds=st.mean(r['loop_seconds_including_audit'] for r in cases),mean_audit_seconds=st.mean(r['audit_seconds'] for r in cases),
        mean_non_audit_loop_seconds=st.mean(r['loop_seconds_excluding_audit_kernels'] for r in cases),
        mean_forward_examples=st.mean(r['learner_costs']['forward_examples'] for r in cases),mean_audit_forward_examples=st.mean(r['audit_forward_examples'] for r in cases),
        mean_stored_tensor_bytes=st.mean(r['learner_costs']['stored_tensor_bytes'] for r in cases),module_counts=[r['learner_costs']['module_count'] for r in cases]))
report=['# E24: no harmful accepted merge observed, but rare answers still deteriorate','',
'All 12 cases completed: four fresh worlds paired across contextual replay, local replay with merging disabled, and unchanged E6 guarded merging, at a two-module cap. Each stream first teaches rare inputs frequently for 512 updates, then reduces their probability to 1/512 while common input means change. The conditional functions stay fixed. Every learner consumes 262,144 observations and persists for 8,192 updates; no oracle region or boundary enters training.', '',
'This experiment targets retained answers outside the samples used to approve compression. Its independent probes are never supplied to the learner. It does not establish general semantic memory or exact language-fact recall.', '',
'| Arm | Mean prequential MSE | Rare MSE after acquisition | Rare MSE at end | Final rare competent worlds | Module counts |', '|---|---:|---:|---:|---:|---|']
for r in summary:report.append(f"| {r['arm']} | {r['mean_prequential_mse']:.6f} | {r['mean_rare_acquisition']:.6f} | {r['mean_rare_final']:.6f} | {r['final_rare_competent']}/4 | {r['module_counts']} |")
report+=['','## The intended merge falsification did not occur','',
'| Arm | Checks | Competent outgoing sources | Accepted merges | Material proposed rare losses | Accepted material losses |', '|---|---:|---:|---:|---:|---:|']
for r in summary:report.append(f"| {r['arm']} | {r['checks']} | {r['source_competent_checks']} | {r['accepted']} | {r['material_proposal_losses']} | {r['accepted_material_losses']} |")
report+=['',
'Guarded sharing accepts ten merges, all from sources competent on the held-out rare probes. None crosses the frozen material-loss rule: source rare MSE at most 0.05, replacement rare MSE greater than max(0.05, twice source error). This is a real negative result for the attempted falsification, not a lack of acquired competence or an inactive merge mechanism. It does not prove arbitrary future-query preservation. Smaller losses, accumulation and inputs outside these finite probes remain possible.', '',
'The non-merging arm proposes twenty replacements with material rare losses, all in one world, but never applies a merge by design. These are not observed erasures. They also do not show that the E6 anchor threshold would necessarily accept those proposals, since that arm deliberately disables acceptance. All checks remain in the complete record.', '',
'## Retained knowledge and operational access separate','',
'| Seed | Arm | Final operational rare MSE | Best stored-module rare MSE (diagnostic) |', '|---:|---|---:|---:|']
for r in rs:
    e=r['endpoints'][-1];best=e['best_persistent_module_diagnostic'];report.append(f"| {r['seed']} | {r['arm']} | {e['operational']['rare']:.6f} | {best['rare']:.6f} |" if best else f"| {r['seed']} | {r['arm']} | {e['operational']['rare']:.6f} | — |")
guard=next(r for r in summary if r['arm']=='guarded_sharing');raw=next(r for r in summary if r['arm']=='context_replay');nomerge=next(r for r in summary if r['arm']=='local_replay_isolation')
report+=['',f'Every arm first acquires rare-query competence in every world. Guarded sharing ends with {guard["mean_rare_final"]/guard["mean_rare_acquisition"]:.2f} times its acquisition error, and three of four final errors narrowly exceed the frozen 0.05 competence threshold. Yet its mean final rare error is {1-guard["mean_rare_final"]/raw["mean_rare_final"]:.1%} lower than context replay and {1-guard["mean_rare_final"]/nomerge["mean_rare_final"]:.1%} lower than non-merging operational predictions. The record must preserve both relative benefit and absolute deterioration.', '',
'In three of four non-merging worlds, a persistent module remains rare-competent while the active operational choice is not. This directly establishes an access/selection gap on these queries. The best-module value uses audit targets to choose a model and is explicitly privileged; it is not reported as an agent capability. The existing router selects an active model using observed batch outcomes, while the rare region can be identified from the current input. A future input-dependent retrieval mechanism should be tested before attributing this gap to absent capacity.', '',
'Rare error can rise through repeated ordinary adaptation, subthreshold merge losses, or changing model selection. These endpoints and immediate merge checks do not isolate every later cause. The final checkpoint audit records how many rare examples remain in each module reservoir. Coverage correlations alone are not a causal test of a sampling repair.', '',
'## Resource accounting and validation','',
'| Arm | Mean operational forward examples | Mean audit forward examples | Mean loop seconds, including audit | Mean audit seconds | Mean counted tensor bytes |', '|---|---:|---:|---:|---:|---:|']
for r in summary:report.append(f"| {r['arm']} | {r['mean_forward_examples']:,.0f} | {r['mean_audit_forward_examples']:,.0f} | {r['mean_loop_seconds']:.3f} | {r['mean_audit_seconds']:.3f} | {r['mean_stored_tensor_bytes']:,.0f} |")
report+=['',
'The lower byte count versus contextual replay partly reflects its smaller configured replay budget. Guarded sharing and the non-merging control use the same per-module anchor budget and cap, but their learned trajectories and final module counts differ. Lower bytes alone do not certify preservation.', '',
'The audit transiently holds the outgoing model and 24,576 bytes of probe inputs/targets. Its model evaluations and timing are recorded separately and included in total study loop work. Parameter bytes of each outgoing model are recorded, but this is not complete peak RAM. Subtracting audit kernels does not remove every Python instrumentation cost. Checkpoint size/serialization time, data-generation time and inherited tensor/parameter peaks remain in the complete output. Full process RAM, energy, complete FLOPs and deployment costs are unmeasured.', '',
'Preflight exercises accepted merges and verifies that audited and unaudited predictions, final weights, anchors, counters and RNG are exactly equal. It checks outgoing models remain unchanged on merge steps, source/candidate mapping, prefix-stable data and exact read-only checkpoint restoration. All twelve scored final checkpoints match frozen source identity and counters, have 8,192 updates and finite weights, and reproduce every recorded final probe prediction exactly. The module/anchor bounds hold.', '',
'No new memory policy or reasoning component was introduced. E24 supplies a scoped pass for immediate accepted-merge safety on the registered probes, a failure of sustained rare-query competence, and a concrete retrieval gap. It neither certifies compression nor establishes efficient general continual learning. The next intervention should distinguish input-dependent access to already retained knowledge from actual information loss, using only observed memory records to train any router.']
for n in ('complete','preflight'):shutil.copyfile(RUNS/f'{n}.json',HERE/f'results/e24_{n}.json')
(HERE/'results/e24_audit.json').write_text(json.dumps(dict(identity=d['identity'],checks=checks),indent=2)+'\n')
(HERE/'results/e24_summary.json').write_text(json.dumps(summary,indent=2)+'\n');(HERE/'E24-RESULT.md').write_text('\n'.join(report)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.5));colors=['#777777','#0072B2','#D55E00']
for i,arm in enumerate(spec['arms']):
    cases=[r for r in rs if r['arm']==arm];a=np.array([[e['operational']['rare'] for e in r['endpoints']] for r in cases]);axes[0].plot(np.arange(1,17)*512,a.mean(0),label=arm,color=colors[i])
nom=[r for r in rs if r['arm']=='local_replay_isolation'];x=np.arange(4)
axes[1].bar(x-.18,[r['endpoints'][-1]['operational']['rare'] for r in nom],.36,label='Operational active model',color='#0072B2')
axes[1].bar(x+.18,[r['endpoints'][-1]['best_persistent_module_diagnostic']['rare'] for r in nom],.36,label='Best stored model (privileged diagnostic)',color='#009E73')
for ax in axes:ax.axhline(.05,color='black',ls=':',lw=1,label='Frozen competence threshold');ax.set_ylabel('Held-out rare-query MSE');ax.legend(fontsize=8)
axes[0].set_xlabel('Observed update (rare probability falls after 512)');axes[0].set_title('Rare answers deteriorate despite low stream error')
axes[1].set_xticks(x,[r['seed'] for r in nom]);axes[1].set_xlabel('Fresh world seed, non-merging control');axes[1].set_title('Knowledge can remain in an unselected model')
fig.suptitle('E24 • 12 complete cases • 0 material rare losses in 10 accepted merges');fig.tight_layout();fig.savefig(HERE/'results/e24_rare_memory.png',dpi=160)
print('\n'.join(report[:25]));print('AUDIT_OK',len(checks))
