"""P1a: charged active acquisition; private truths never enter policy state."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def entropy(p):
    return -p * np.log(p) - (1-p) * np.log1p(-p)


def information_gain(a, b, harmonic):
    # E[H(Bernoulli(theta))] = H_(a+b) - (a H_a + b H_b)/(a+b).
    n = a+b
    return entropy(a/n) - harmonic[n] + (a*harmonic[a]+b*harmonic[b])/n


class Learner:
    def __init__(self, seeds, sources, steps, policy, window):
        self.a = np.ones((seeds, sources), dtype=np.int64)
        self.b = self.a.copy()
        self.count = np.zeros_like(self.a)
        self.surprise = np.full(self.a.shape, np.log(2.))
        self.harmonic = np.r_[0., np.cumsum(1 / np.arange(1, steps+3))]
        self.policy, self.window = policy, window
        self.history = np.zeros((seeds, sources, window), dtype=np.int8)
        self.rows = np.arange(seeds)
        self.scores = np.zeros_like(self.surprise)
        self.refresh(np.zeros(seeds, dtype=int))
        if policy == 'entropy':
            self.scores[:] = np.log(2.)
        elif policy == 'information_gain':
            self.scores[:] = np.log(2.) - .5
        elif policy == 'surprise':
            self.scores[:] = np.log(2.)

    def predictions(self):
        return self.a / (self.a+self.b)

    def refresh(self, actions):
        ix = self.rows, actions
        a, b = self.a[ix], self.b[ix]
        if self.policy == 'entropy':
            self.scores[ix] = entropy(a/(a+b))
        elif self.policy == 'information_gain':
            self.scores[ix] = information_gain(a, b, self.harmonic)
        elif self.policy == 'surprise':
            self.scores[ix] = self.surprise[ix]

    def choose(self, t, explore, random_action, tie_u):
        k = self.a.shape[1]
        if t < 2*k:
            return np.full(len(self.rows), t % k, dtype=int)
        if self.policy == 'uniform':
            return random_action
        tied = self.scores == self.scores.max(axis=1, keepdims=True)
        rank = np.floor(tie_u*tied.sum(1)).astype(int)+1
        selected = (tied.cumsum(1) >= rank[:, None]).argmax(1)
        return np.where(explore < .1, random_action, selected)

    def observe(self, actions, outcomes):
        ix = self.rows, actions
        a, b = self.a[ix], self.b[ix]
        p = a/(a+b)
        loss = -np.where(outcomes, np.log(p), np.log1p(-p))
        self.surprise[ix] = .9*self.surprise[ix] + .1*loss
        if self.window:
            pos = self.count[ix] % self.window
            hix = self.rows, actions, pos
            old = self.history[hix]
            full = self.count[ix] >= self.window
            self.a[ix] -= full*old
            self.b[ix] -= full*(1-old)
            self.history[hix] = outcomes
        self.a[ix] += outcomes
        self.b[ix] += 1-outcomes
        self.count[ix] += 1
        self.refresh(actions)

    def bytes(self):
        return sum(x.nbytes for x in vars(self).values() if isinstance(x, np.ndarray))


def preflight():
    harmonic = np.r_[0., np.cumsum(1/np.arange(1, 2049))]
    # Independent exact expected posterior-entropy decrease, using numerical
    # integration of Beta density on a fine grid; no policy evaluation involved.
    x = (np.arange(200000)+.5)/200000
    errors = []
    for a,b in [(1,1),(2,7),(8,3),(20,20)]:
        density = x**(a-1)*(1-x)**(b-1)
        density /= density.sum()
        numerical = entropy(a/(a+b)) - (density*entropy(x)).sum()
        errors.append(abs(numerical-information_gain(a,b,harmonic)))
    assert max(errors) < 1e-8
    learner = Learner(1, 2, 100, 'information_gain', 3)
    seen = []
    for y in [1,0,1,1,0,0,1]:
        seen.append(y)
        learner.observe(np.array([0]), np.array([y]))
        assert learner.a[0,0] == 1+sum(seen[-3:])
        assert learner.b[0,0] == 1+len(seen[-3:])-sum(seen[-3:])
        assert learner.a[0,1] == learner.b[0,1] == 1
    return dict(integration_max_error=max(errors), rolling_updates_exact=7,
                equal_entropy=float(np.log(2)),
                ig_a1_b1=float(information_gain(1,1,harmonic)),
                ig_a512_b512=float(information_gain(512,512,harmonic)))


def execute(out, seeds, steps):
    started = time.perf_counter()
    k, s = 32, len(seeds)
    rows = np.arange(s)
    truths, masks = [], []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        bits = rng.integers(0,2,k).astype(float)
        mask = np.zeros(k, dtype=bool)
        mask[rng.permutation(k)[:k//2]] = True
        truths.append(bits)
        masks.append(mask)
    base, noise_mask = np.array(truths), np.array(masks)
    # Common independent potential outcome for every source/time/seed, plus
    # action randomness; policy only receives the selected present observation.
    feedback = np.stack([np.random.default_rng(seed+100000).random((steps,k))
                         for seed in seeds], axis=1)
    draws = np.stack([np.random.default_rng(seed+200000).random((steps,3))
                      for seed in seeds], axis=1)
    data_hash = hashlib.sha256(base.tobytes()+noise_mask.tobytes()+feedback.tobytes()+draws.tobytes()).hexdigest()
    report = dict(seeds=seeds, steps=steps, sources=k, data_sha256=data_hash,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  preflight=preflight(), cases=[])
    for regime in ['deterministic', 'mixed', 'mixed_change']:
        for window in [0,64]:
            for policy in ['uniform','surprise','entropy','information_gain']:
                learner = Learner(s,k,steps,policy,window)
                error = np.zeros(s)
                signal_error = np.zeros(s)
                second_error = np.zeros(s)
                allocations = np.zeros(s, dtype=int)
                # Evaluation arrays are external to Learner and never observed.
                trace = []
                trajectories = [hashlib.sha256() for _ in seeds]
                begin = time.perf_counter()
                for t in range(steps):
                    truth = base.copy()
                    if regime == 'mixed_change' and t >= steps//2:
                        truth = 1-truth
                    signal = np.ones_like(noise_mask) if regime == 'deterministic' else ~noise_mask
                    if regime != 'deterministic':
                        truth[noise_mask] = .5
                    p = learner.predictions()
                    se = (p-truth)**2
                    all_err = se.mean(1)
                    error += all_err
                    signal_error += (se*signal).sum(1)/signal.sum(1)
                    if t >= steps//2:
                        second_error += all_err
                    action = learner.choose(t, draws[t,:,0], (draws[t,:,1]*k).astype(int), draws[t,:,2])
                    outcome = (feedback[t, rows, action] < truth[rows,action]).astype(int)
                    for j in range(s):
                        trajectories[j].update(bytes([int(action[j]), int(outcome[j])]))
                    allocations += ~signal[rows,action]
                    learner.observe(action, outcome)
                    if t % 64 == 0:
                        trace.append(dict(step=t, excess_brier=all_err.tolist()))
                elapsed = time.perf_counter()-begin
                endpoint = ((learner.predictions()-truth)**2).mean(1)
                case = dict(regime=regime, window=window, policy=policy,
                            mean_excess_brier=(error/steps).tolist(),
                            signal_excess_brier=(signal_error/steps).tolist(),
                            second_half_excess_brier=(second_error/(steps//2)).tolist(),
                            endpoint_excess_brier=endpoint.tolist(),
                            noise_observation_fraction=(allocations/steps).tolist(),
                            observations_per_seed=learner.count.sum(1).tolist(),
                            counts_by_seed_source=learner.count.tolist(),
                            policy_and_eval_seconds=elapsed,
                            persistent_array_bytes_all_seeds=learner.bytes(),
                            trajectory_sha256_by_seed=[h.hexdigest() for h in trajectories],
                            posterior_updates=steps*s,
                            score_entries_recomputed=steps*s if policy!='uniform' else 0,
                            selection_score_entries_scanned=(steps-2*k)*s*k if policy!='uniform' else 0,
                            diagnostic_trace=trace)
                assert case['observations_per_seed'] == [steps]*s
                report['cases'].append(case)
                print(json.dumps({key:case[key] for key in ['regime','window','policy','policy_and_eval_seconds']}), flush=True)
                out.write_text(json.dumps(report, indent=2)+'\n')
    report['total_seconds'] = time.perf_counter()-started
    report['status'] = 'complete'
    out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    execute(args.out, [36901,36902] if args.development else list(range(37001,37025)),
            256 if args.development else 4096)
