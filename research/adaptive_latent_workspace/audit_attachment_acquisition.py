"""Replay all scored choices; reconstruct posterior counts from raw histories.

Reuses the frozen choice/update code to verify exact replay hashes, but separately
checks every choice's allowed support, every selected posterior, and saved losses.
Does not claim independent reimplementation of the entire experiment.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from probe_attachment_acquisition import Learner


def audit(path, out):
    start = time.perf_counter()
    record = json.loads(path.read_text())
    assert record['status'] == 'complete'
    implementation = Path(__file__).with_name('probe_attachment_acquisition.py')
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == record['source_sha256']
    seeds, steps, k = record['seeds'], record['steps'], record['sources']
    n = len(seeds)
    rows = np.arange(n)
    bits, noisy = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bits.append(rng.integers(0,2,k).astype(float))
        mask = np.zeros(k,dtype=bool)
        mask[rng.permutation(k)[:k//2]] = True
        noisy.append(mask)
    base, noise = np.array(bits), np.array(noisy)
    feedback = np.stack([np.random.default_rng(seed+100000).random((steps,k)) for seed in seeds],axis=1)
    draws = np.stack([np.random.default_rng(seed+200000).random((steps,3)) for seed in seeds],axis=1)
    assert hashlib.sha256(base.tobytes()+noise.tobytes()+feedback.tobytes()+draws.tobytes()).hexdigest()==record['data_sha256']
    checked_updates = checked_snapshots = 0
    max_metric_error = 0.
    for case in record['cases']:
        policy, window, regime = case['policy'], case['window'], case['regime']
        learner = Learner(n,k,steps,policy,window)
        histories = [[deque(maxlen=window or None) for _ in range(k)] for _ in seeds]
        hashes = [hashlib.sha256() for _ in seeds]
        loss = np.zeros(n)
        second = np.zeros(n)
        signal_loss = np.zeros(n)
        counts = np.zeros((n,k),dtype=int)
        noise_count = np.zeros(n,dtype=int)
        for t in range(steps):
            truth = 1-base if regime=='mixed_change' and t>=steps//2 else base.copy()
            signal = np.ones_like(noise) if regime=='deterministic' else ~noise
            if regime!='deterministic':
                truth[noise] = .5
            # At regular complete snapshots, derive ALL posteriors only from
            # recorded outcomes, independently of mutable learner count arrays.
            if t % 64 == 0:
                successes = np.array([[sum(h) for h in row] for row in histories])
                lengths = np.array([[len(h) for h in row] for row in histories])
                assert np.array_equal(learner.a, 1+successes)
                assert np.array_equal(learner.b, 1+lengths-successes)
                checked_snapshots += n*k
            prediction = learner.predictions()
            squared = (prediction-truth)**2
            loss += squared.mean(1)
            signal_loss += (squared*signal).sum(1)/signal.sum(1)
            if t>=steps//2:
                second += squared.mean(1)
            action = learner.choose(t,draws[t,:,0],(draws[t,:,1]*k).astype(int),draws[t,:,2])
            if t>=2*k and policy!='uniform':
                not_random = draws[t,:,0]>=.1
                assert np.array_equal(learner.scores[rows,action][not_random],learner.scores.max(1)[not_random])
            if t>=2*k:
                forced_random = np.ones(n,dtype=bool) if policy=='uniform' else draws[t,:,0]<.1
                assert np.array_equal(action[forced_random],(draws[t,:,1]*k).astype(int)[forced_random])
            observed = (feedback[t,rows,action]<truth[rows,action]).astype(int)
            learner.observe(action,observed)
            for j,(a,y) in enumerate(zip(action,observed)):
                h = histories[j][a]
                h.append(int(y))
                assert learner.a[j,a] == 1+sum(h)
                assert learner.b[j,a] == 1+len(h)-sum(h)
                hashes[j].update(bytes([int(a),int(y)]))
                counts[j,a] += 1
            checked_updates += n
            noise_count += ~signal[rows,action]
        assert [h.hexdigest() for h in hashes] == case['trajectory_sha256_by_seed']
        assert counts.tolist() == case['counts_by_seed_source']
        metrics = dict(mean_excess_brier=loss/steps,
                       signal_excess_brier=signal_loss/steps,
                       second_half_excess_brier=second/(steps//2),
                       endpoint_excess_brier=((learner.predictions()-truth)**2).mean(1),
                       noise_observation_fraction=noise_count/steps)
        for key, val in metrics.items():
            diff = float(np.abs(val-np.array(case[key])).max())
            max_metric_error = max(max_metric_error,diff)
            assert diff < 1e-14
        print(regime,window,policy,'AUDITED',flush=True)
    result = dict(status='complete', cases=len(record['cases']),
                  exact_seed_trajectories=len(record['cases'])*n,
                  independently_reconstructed_selected_posteriors=checked_updates,
                  complete_snapshot_source_posteriors=checked_snapshots,
                  max_metric_error=max_metric_error, seconds=time.perf_counter()-start,
                  scored_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    audit(args.input,args.out)
