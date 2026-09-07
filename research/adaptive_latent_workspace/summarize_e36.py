"""Audited E36 action outcomes and censored, pre-decision map acquisition."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def mean(values):
    return float(np.mean(values))


def acquisition(rows,spec):
    """Each real segment is retained, including never-acquired segments.

    Coverage is pre-action all-pair argmax at the current memory/history. A first
    point threshold is not stable acquisition; explicitly retain later failures.
    Unique observed pairs are a coverage proxy, not an information lower bound.
    """
    records=[]
    begin=0
    for index,(duration,slot) in enumerate(zip(spec['durations'],spec['slots'])):
        part=rows[begin:begin+duration]
        assert len(part)==duration
        if part[0]['private_diagnostic']['map_correct'] is None:
            return None
        coverage=[r['private_diagnostic']['map_correct'] for r in part]
        first=next((t for t,c in enumerate(coverage) if c>=11),None)
        seen=set()
        observed_at_first=None
        for t,row in enumerate(part):
            if t==first:
                observed_at_first=len(seen)
            seen.add((row['observation'],row['executed_action']))
        # Independent direct prefix search of the threshold, with no inferred
        # threshold hit from loss or a successful-only aggregate.
        prefix=0
        while prefix<len(part) and part[prefix]['private_diagnostic']['map_correct']<11:
            prefix+=1
        assert prefix==(duration if first is None else first)
        records.append(dict(segment=index,slot=slot,begin=begin,duration=duration,
                            first_threshold_action_offset=first,censored=first is None,
                            restricted_action_fraction=prefix/duration,
                            end_pre_action_map_correct=coverage[-1],
                            mean_map_accuracy=sum(coverage)/(duration*12),
                            later_below_threshold=sum(c<11 for c in coverage[first+1:]) if first is not None else None,
                            segment_unique_observed_pairs=len(seen),
                            observed_pairs_before_first_threshold=observed_at_first))
        begin+=duration
    assert begin==len(rows)
    return records


def run(folder,auditfile,dest):
    complete=json.loads((folder/'complete.json').read_text())
    audit=json.loads(auditfile.read_text())
    assert complete['status']==audit['status']=='complete'
    assert audit['scored_complete_sha256']==hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest()
    data=json.loads((folder/'data.json').read_text())
    spec_by_name={s['name']:s for s in data['evaluation']}
    cases=[]
    grouped=defaultdict(list)
    for metadata in complete['evaluation']:
        raw=json.loads(Path(metadata['result_file']).read_text())
        condition=raw['condition'] if isinstance(raw['condition'],str) else raw['condition']['name']
        spec=spec_by_name[raw['world']]
        records=acquisition(raw['rows'],spec)
        case=dict(seed=metadata.get('seed'),condition=condition,world=raw['world'],regime=spec['regime'],
                  actions=raw['actions'],goals=raw['goals'],accuracy=raw['correct']/raw['actions'],
                  mean_nll=raw['mean_nll'],acting_seconds=raw['acting_seconds'],total_seconds=raw['total_seconds'],
                  work=raw['work'],storage=raw['storage'],acquisition=records,
                  late_goals=sum(r['reward'] for r in raw['rows'][len(raw['rows'])//2:]),
                  source_file=metadata['result_file'])
        cases.append(case)
        grouped[condition,spec['regime']].append(case)
    aggregates=[]
    for (condition,regime),group in grouped.items():
        rec=[r for c in group for r in (c['acquisition'] or [])]
        aggregates.append(dict(condition=condition,regime=regime,cases=len(group),
                               worlds=len(set(c['world'] for c in group)),
                               mean_goals=mean([c['goals'] for c in group]),
                               min_goals=min(c['goals'] for c in group),max_goals=max(c['goals'] for c in group),
                               mean_late_goals=mean([c['late_goals'] for c in group]),
                               mean_actual_accuracy=mean([c['accuracy'] for c in group]),
                               mean_nll=mean([c['mean_nll'] for c in group]) if group[0]['mean_nll'] is not None else None,
                               acting_seconds=sum(c['acting_seconds'] for c in group),
                               total_seconds=sum(c['total_seconds'] for c in group),
                               acquired_segments=sum(not r['censored'] for r in rec),segments=len(rec),
                               mean_restricted_action_fraction=mean([r['restricted_action_fraction'] for r in rec]) if rec else None,
                               threshold_hits_with_later_failure=sum(r['later_below_threshold'] is not None and r['later_below_threshold']>0 for r in rec)))
    indexed={(c['seed'],c['condition'],c['world']):c for c in cases}
    paired=[]
    for c in cases:
        if c['condition']!='e2e_ttt':
            continue
        for baseline in ['static_ttt','first_order_ttt','e2e_frozen','bfs','counts','random']:
            other=indexed[c['seed'] if baseline not in ['bfs','counts','random'] else None,baseline,c['world']]
            paired.append(dict(seed=c['seed'],world=c['world'],regime=c['regime'],baseline=baseline,
                               goal_difference=c['goals']-other['goals'],accuracy_difference=c['accuracy']-other['accuracy'],
                               nll_difference=c['mean_nll']-other['mean_nll'] if other['mean_nll'] is not None else None,
                               acting_time_ratio=c['acting_seconds']/other['acting_seconds']))
    noise=complete['manifest']['protocol']['evaluation_noise_rate']
    n=complete['manifest']['protocol']['states']
    best=1-noise+noise/n
    entropy=-best*math.log(best)-(n-1)*(noise/n)*math.log(noise/n)
    summary=dict(status='complete',source_commit=complete['manifest']['commit'],development=complete['manifest']['development'],
                 audit=audit,training=complete['training'],cases=cases,aggregates=aggregates,paired=paired,
                 scored_seconds=complete['seconds'],process_max_rss_kib=complete['process_max_rss_kib'],
                 noisy_perfect_map_expected_accuracy=best,noisy_perfect_map_expected_nll=entropy,
                 input_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest())
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# E36 audited action results', '',
           '**Development-only report. Do not infer capability.**' if summary['development'] else 'Source-frozen scored run; no checkpoint or hyperparameter selection on these worlds.', '',
           'The planner is fixed. This tests whether predictive continual learning supplies',
           'useful dynamics for actual actions. It does not train the goal generator or',
           'reasoning procedure, nor prove memory compression or recursive improvement.', '',
           '## Goal completion', '',
           'Each world has a fixed paid-action budget. Neural rows cross initialization',
           'seeds with the same worlds; control rows run once per world. Repeating a',
           'control for paired neural comparisons does not create independent worlds.',
           'Evaluation evidence is action-dependent, unlike equal-data training.', '',
           '| Condition | Regime | Cases / worlds | Mean goals | Range | Mean actual accuracy | Mean NLL |',
           '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregates:
        nll='—' if a['mean_nll'] is None else f"{a['mean_nll']:.4f}"
        lines.append(f"| {a['condition']} | {a['regime']} | {a['cases']} / {a['worlds']} | {a['mean_goals']:.2f} | {a['min_goals']}–{a['max_goals']} | {a['mean_actual_accuracy']:.2%} | {nll} |")
    lines+=['','## E2E paired differences','',
            'Positive goal differences favor E2E TTT. Counts are descriptive paired',
            'cases, not independent population trials or significance tests.','',
            '| Baseline | Regime | Better / tied / worse goals | Mean goal difference | Mean acting-time ratio |',
            '|---|---|---:|---:|---:|']
    by_pair=defaultdict(list)
    for p in paired:
        by_pair[p['baseline'],p['regime']].append(p)
    for (baseline,regime),group in by_pair.items():
        ds=[p['goal_difference'] for p in group]
        lines.append(f"| {baseline} | {regime} | {sum(d>0 for d in ds)} / {sum(d==0 for d in ds)} / {sum(d<0 for d in ds)} | {mean(ds):+.2f} | {mean([p['acting_time_ratio'] for p in group]):.2f} |")
    lines+=['','## Acquisition and failure after first acquisition','',
            'Threshold: at least 11 of 12 transition argmaxes correct at a pre-action',
            'probe. Probes reuse already-paid model queries; labels are evaluator-only.',
            'This measures map knowledge, not actual noisy-outcome accuracy. Time starts',
            'at each hidden segment boundary. Zero means threshold held before any new',
            'segment action. A never-hit segment is censored at its full duration.',
            'The first point hit is not stable acquisition; later failures are retained.',
            'No post-final-action query or per-recovery wall-time measurement is available.','',
            '| Condition | Regime | Acquired / all segments | Mean restricted action fraction | Hits followed by failure |',
            '|---|---|---:|---:|---:|']
    for a in aggregates:
        if a['segments']:
            lines.append(f"| {a['condition']} | {a['regime']} | {a['acquired_segments']} / {a['segments']} | {a['mean_restricted_action_fraction']:.3f} | {a['threshold_hits_with_later_failure']} |")
    lines+=['','## Costs and scope','',
            f"Scorer total time: {complete['seconds']:.3f}s. Audit: {audit['seconds']:.3f}s.",
            f"Audit verifies {audit['training_updates']} outer updates, {audit['evaluation_actions']} evaluation actions and {audit['exact_checkpoints']} noninitial checkpoints.",
            'Training replay reuses the training routine. Evaluation gradient updates',
            'bypass the action wrapper; physical transitions/noise/goals and NumPy',
            'planner probabilities are independently reconstructed. Source hashes,',
            'saved logits, optimizer/model states and declared work are checked.', '',
            'Training and acting seconds, operation counts and storage are preserved per',
            'case in summary.json. Acting timers include environment stepping and Python',
            'control/update overhead, but exclude private map diagnostics and checkpoint',
            'serialization. Total case timers include those. Neither is isolated kernel',
            'latency. Reported state bytes omit Python/allocator overhead. Process peak',
            'RSS includes multiple models, optimizer states, training graphs and data.', '',
            f"With the declared noise, perfect map knowledge yields expected actual accuracy {best:.2%} and NLL {entropy:.6f}.",
            'These are distributional expectations, not bounds on each finite realized',
            'sample. The controller is not supplied the true noise rate or map.', '',
            'These randomly generated finite cycles contain strong structure, a fixed',
            'alphabet, and only two test worlds per regime in the scored protocol.',
            'Neither a local gain nor failure identifies a universal AI improvement-rate',
            'bound. Use goal outcomes, retained knowledge and costs together before',
            'deciding the next architecture change. The full goal remains active.','']
    (dest/'tables.md').write_text('\n'.join(lines))
    print('E36_SUMMARIZED',len(cases),sum(len(c['acquisition'] or []) for c in cases),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    run(a.input,a.audit,a.out)
