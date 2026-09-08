"""Post hoc E36 controller intervention, with unchanged trained models/updates.

Run-local function substitution reuses the frozen evaluator and its trajectory
auditor. The substitution and independent policy implementation are hashed.
The inherited E36 manifest describes the input; intervention.json describes
the actual supplemental experiment. Never treat this as the original protocol.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

import audit_e36 as audit
import e36_action_learning as original
from action_planning import goal_values
from e2e_core import Config, E2ECore


TIE_TOLERANCE = 1e-6


def greedy(probabilities, observation, config):
    q = goal_values(probabilities.double(), observation.goal, config)[observation.state]
    best = (q.max() - q <= TIE_TOLERANCE).double()
    return (1-config.exploration)*best/best.sum()+config.exploration/len(q)


def independent_greedy(scores, current, goal, protocol):
    # Reproduce the input float32 normalization, then independently perform
    # explicit float64 Bellman sums, without calling the Torch planner.
    x = np.asarray(scores, dtype=np.float32)
    p = np.exp(x-x.max(axis=-1, keepdims=True))
    p = (p/p.sum(axis=-1, keepdims=True)).astype(np.float64)
    n, a, cfg = protocol['states'], protocol['actions'], protocol['planning']
    value = np.zeros(n)
    for _ in range(cfg['horizon']):
        q = np.array([[sum(p[s,j,y]*(1. if y==goal else cfg['discount']*value[y])
                           for y in range(n)) for j in range(a)] for s in range(n)])
        value = q.max(axis=1)
    best = (q[current].max()-q[current] <= TIE_TOLERANCE).astype(float)
    return (1-cfg['exploration'])*best/best.sum()+cfg['exploration']/a


def run(folder, out, training_audit, preflight=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not out.exists()
    complete = json.loads((folder/'complete.json').read_text())
    manifest = complete['manifest']
    data = json.loads((folder/'data.json').read_text())
    root = Path(__file__).parent
    for f, sha in manifest['sources'].items():
        assert hashlib.sha256((root/f).read_bytes()).hexdigest() == sha
    verified = json.loads(training_audit.read_text())
    assert verified['status'] == 'complete_training'
    for path, sha in verified['final_checkpoint_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (out/'data.json').symlink_to(folder/'data.json')
    for entry in complete['training']:
        name = f"{entry['seed']}-{entry['arm']}"
        (out/name).symlink_to(folder/name, target_is_directory=True)
    note = dict(scope='post hoc frozen-model controller intervention; not fresh generalization',
                preflight=preflight, policy='epsilon greedy, uniform ties within absolute Q tolerance',
                tie_tolerance=TIE_TOLERANCE, exploration=manifest['protocol']['planning']['exploration'],
                parent_complete_sha256=hashlib.sha256((folder/'complete.json').read_bytes()).hexdigest(),
                own_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                audit_source_sha256=hashlib.sha256((root/'audit_e36.py').read_bytes()).hexdigest(),
                training_audit_sha256=hashlib.sha256(training_audit.read_bytes()).hexdigest(),
                planned_cases=4 if preflight else 80,
                claim='Test whether original soft action selection limits these learned and count models. No retraining or private maps enter decisions.')
    (out/'intervention.json').write_text(json.dumps(note, indent=2)+'\n')
    original.plan = greedy
    audit.plan = greedy
    audit.independent_plan = independent_greedy
    protocol = manifest['protocol']
    specs = data['evaluation'][:1] if preflight else data['evaluation']
    seeds = protocol['seeds'][:1] if preflight else protocol['seeds']
    results = []
    started = time.perf_counter()
    for seed in seeds:
        for condition in protocol['evaluation_arms']:
            if condition['update']=='frozen':
                continue
            model = E2ECore(Config(**protocol['config']), seed=0)
            checkpoint = folder/f"{seed}-{condition['training']}"/f"training-{protocol['outer_steps']}.pt"
            model.load_state_dict(torch.load(checkpoint, weights_only=False)['model'])
            for spec in specs:
                target = out/f"eval-{seed}-{condition['name']}-{spec['name']}"
                result = original.evaluate(model, spec, condition, protocol, target)
                result['seed'] = seed
                results.append(result)
                (out/'progress.json').write_text(json.dumps(results)+'\n')
                print('GREEDY_CASE', seed, condition['name'], spec['name'], result['goals'], flush=True)
    for spec in specs:
        results.append(original.evaluate(None, spec, {}, protocol, out/f"control-counts-{spec['name']}", control='counts'))
    assert len(results)==note['planned_cases']
    result = dict(status='complete', manifest=manifest, training=complete['training'], evaluation=results,
                  intervention=note, seconds=time.perf_counter()-started)
    (out/'complete.json').write_text(json.dumps(result, indent=2)+'\n')
    # Main physics/update/checkpoint auditor, with a distinct NumPy greedy policy.
    # Training checks are inherited via verified hashes, not re-executed.
    audit.run(out, out/'audit.json', stage='evaluation', training_audit=training_audit)
    print('GREEDY_INTERVENTION_AUDITED', len(results), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--training-audit', type=Path, required=True)
    p.add_argument('--preflight', action='store_true')
    a=p.parse_args()
    run(a.input, a.out, a.training_audit, a.preflight)
