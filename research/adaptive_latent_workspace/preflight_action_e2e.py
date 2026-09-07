"""Real/imagined provenance, dense-reference and meta-gradient checks.

Development fixtures only: no acquisition, task-performance or efficiency claim.
"""
import argparse
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import time

import torch

from action_e2e import ActionE2E
from e2e_core import Config, E2ECore
from preflight_e2e_core import assert_state_equal, dense_reference


class ExecutionFixture:
    """Hidden port destinations, solely for checking actual-action receipts."""
    def __init__(self):
        self.transitions = ((1, 3), (2, 0), (3, 1), (0, 2))
        self.observation = 0

    def step(self, action):
        self.observation = self.transitions[self.observation][action]
        return self.observation


def assert_action_state_equal(a, b):
    assert a.observation == b.observation and a.real_steps == b.real_steps
    assert_state_equal(a.stream, b.stream)


def rejected(call, error):
    try:
        call()
    except error:
        return True
    raise AssertionError("Expected invalid operation to be rejected")


def run(out):
    start = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    cfg = Config(vocab=6, width=16, hidden=24, layers=3, suffix=1,
                 heads=2, window=4, chunk=2, inner_lr=.3, clip=1., init_std=.15)
    checks = {}
    core = E2ECore(cfg, seed=36001, dtype=torch.float64)
    adapter = ActionE2E(core, states=4, actions=2)
    initial = {k: p.detach().clone() for k, p in core.p.items()}
    actual_actions = [0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1]
    fixture = ExecutionFixture()
    trajectory = []
    for action in actual_actions:
        before = fixture.observation
        outcome = fixture.step(action)
        advisory = 1 - action
        assert outcome != fixture.transitions[before][advisory]
        trajectory.append((before, action, outcome, advisory))

    # Full-sequence frozen reference, including cache wraparound and final-only
    # prediction supervision. No wrapper forecast/observe is used by reference.
    tokens = torch.tensor([v for before, action, _, _ in trajectory
                           for v in (before, adapter.states + action)])
    targets = torch.tensor([row[2] for row in trajectory])
    dense = dense_reference(core, tokens)[1::2, :adapter.states]
    expected_loss = dense.logsumexp(-1) - dense[torch.arange(len(targets)), targets]
    state = adapter.initial_state(0)
    losses, predictions = [], []
    for before, action, outcome, advisory in trajectory:
        assert before == state.observation
        forecast = adapter.forecast(state, action)
        loss, state, work, receipt = adapter.observe(forecast, outcome, mode="frozen")
        assert receipt == dict(step=state.real_steps, observation=before,
                               executed_action=action, outcome=outcome,
                               provenance="external_observation")
        assert receipt["executed_action"] != advisory
        assert work["forward_tokens"] == 2 and work["inner_updates"] == 0
        predictions.append(forecast.logits)
        losses.append(loss)
    torch.testing.assert_close(torch.stack(predictions), dense, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(torch.stack(losses), expected_loss, atol=2e-12, rtol=2e-12)
    checks["frozen_dense_max_logit_error"] = float((torch.stack(predictions)-dense).abs().max().detach())
    checks["executed_not_advisory_receipts"] = len(trajectory)

    # Intervention on the new observation changes learning, never the earlier
    # prediction or its consumed prefix. These are two possible real histories.
    origin = adapter.initial_state(0).detached()
    forecast = adapter.forecast(origin, 0)
    prediction = forecast.logits.detach().clone()
    _, a, _, _ = adapter.observe(forecast, 1, mode="first_order")
    _, b, _, _ = adapter.observe(forecast, 3, mode="first_order")
    assert torch.equal(prediction, forecast.logits)
    assert any(not torch.equal(a.stream.fast[k], b.stream.fast[k]) for k in core.fast_names)
    assert a.stream.position == b.stream.position == 2
    for aa, bb in zip(a.stream.cache, b.stream.cache):
        for x, y in zip(aa, bb):
            assert torch.equal(x, y)
    checks["outcome_intervention_preserves_prior_logits_and_cache"] = True

    # Save real state, spend several model calls on abandoned hypothetical paths,
    # and then execute the same real action. Every discarded branch is charged.
    state = a.detached()
    before = adapter.restore(adapter.payload(state))
    branches = [adapter.imagine(state, action) for action in range(adapter.actions)]
    branches += [adapter.imagine(branches[0].branch, action) for action in range(adapter.actions)]
    branches.append(adapter.imagine(branches[-1].branch, 0))
    assert_action_state_equal(state, before)
    assert sum(x.work["forward_tokens"] for x in branches) == 10
    assert sum(x.work["inner_updates"] for x in branches) == 0
    for imagined in branches:
        rejected(lambda: adapter.observe(imagined, 0), TypeError)
        rejected(lambda: adapter.forecast(imagined.branch, 0), TypeError)
        rejected(lambda: adapter.payload(imagined.branch), TypeError)
    f1, f2 = adapter.forecast(state, 1), adapter.forecast(before, 1)
    assert torch.equal(f1.logits, f2.logits)
    _, s1, _, _ = adapter.observe(f1, 0, mode="first_order")
    _, s2, _, _ = adapter.observe(f2, 0, mode="first_order")
    assert_action_state_equal(s1, s2)
    checks["abandoned_imagination_real_continuation_exact"] = True
    checks["imagined_forward_tokens_charged"] = 10
    checks["imagined_labels_and_persistence_rejected"] = len(branches)

    # Actual serialization midway through an adaptive real trajectory. The
    # outcome symbol and executed-step counter are included with weights and KV.
    def continuation(state, rows):
        scores, receipts, work = [], [], []
        for before, action, outcome, _ in rows:
            assert before == state.observation
            forecast = adapter.forecast(state, action)
            _, state, cost, receipt = adapter.observe(forecast, outcome, mode="first_order")
            scores.append(forecast.logits.detach())
            receipts.append(receipt)
            work.append(cost)
            state = state.detached()
        return state, scores, receipts, work

    full = continuation(adapter.initial_state(0).detached(), trajectory)
    prefix = continuation(adapter.initial_state(0).detached(), trajectory[:5])
    buffer = io.BytesIO()
    torch.save(adapter.payload(prefix[0]), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    resumed = continuation(adapter.restore(saved), trajectory[5:])
    assert_action_state_equal(full[0], resumed[0])
    assert torch.equal(torch.stack(full[1]), torch.stack(prefix[1] + resumed[1]))
    assert full[2] == prefix[2] + resumed[2]
    assert full[3] == prefix[3] + resumed[3]
    other = ActionE2E(E2ECore(cfg, seed=36002, dtype=torch.float64), states=4, actions=2)
    rejected(lambda: other.restore(saved), ValueError)
    assert all(torch.equal(initial[k], p) for k, p in core.p.items())
    checks["serialized_adaptive_continuation_exact"] = True
    checks["different_model_rejected_and_base_immutable"] = True
    checks["adaptive_fixture_real_steps"] = len(trajectory)
    checks["adaptive_fixture_forward_tokens"] = sum(w["forward_tokens"] for w in full[3])
    checks["adaptive_fixture_gradient_tokens"] = sum(w["gradient_tokens"] for w in full[3])
    checks["final_persistent_tensor_bytes"] = full[0].stream.bytes()

    # Full second-order directional derivative across four executed actions and
    # observations, including fast-dependent KV with two adaptive layers. This
    # checks an E2E predictive objective, not a differentiated planner or reward.
    fd = []
    rng = torch.Generator().manual_seed(36003)
    for clipping in (100., .001):
        tested = E2ECore(replace(cfg, suffix=2, clip=clipping), seed=36004, dtype=torch.float64)
        learner = ActionE2E(tested, states=4, actions=2)
        def objective(params, mode="second_order"):
            current = learner.initial_state(0, params=params)
            terms, norms = [], []
            for before, action, outcome, _ in trajectory[:4]:
                assert current.observation == before
                forecast = learner.forecast(current, action, params=params)
                loss, current, work, _ = learner.observe(forecast, outcome, mode=mode)
                terms.append(loss)
                norms.append(work["inner_grad_norm"])
            return torch.stack(terms).mean(), norms
        params = dict(tested.p)
        objective_value, norms = objective(params)
        assert all((value < clipping) == (clipping == 100.) for value in norms)
        gradient = torch.autograd.grad(objective_value, tuple(params.values()))
        direction = {k: torch.randn(v.shape, generator=rng, dtype=v.dtype) for k, v in params.items()}
        magnitude = sum(v.square().sum() for v in direction.values()).sqrt()
        direction = {k: v/magnitude for k, v in direction.items()}
        analytic = sum((g*direction[k]).sum() for k, g in zip(params, gradient)).item()
        epsilon = 5e-5
        sides = [objective({k: (v.detach()+sign*epsilon*direction[k]).requires_grad_(True)
                            for k, v in params.items()})[0].item() for sign in (-1, 1)]
        central = (sides[1]-sides[0])/(2*epsilon)
        assert abs(central-analytic) < 2e-7, (central, analytic)
        first_order = torch.autograd.grad(objective(params, "first_order")[0], tuple(params.values()))
        gap = sum((g-f).square().sum() for g, f in zip(gradient, first_order)).sqrt().item()
        assert gap > 1e-7
        fd.append(dict(clip=clipping, inner_norms=norms, analytic=analytic,
                       central_difference=central, absolute_error=abs(central-analytic),
                       full_vs_first_order_gradient_l2=gap))
    checks["two_suffix_action_meta_gradient"] = fd
    checks["range_checks"] = all((rejected(lambda: adapter.initial_state(4), ValueError),
                                  rejected(lambda: adapter.forecast(origin, 2), ValueError),
                                  rejected(lambda: adapter.observe(forecast, -1), ValueError)))
    report = dict(event="ACTION_E2E_PREFLIGHT_PASS", checks=checks,
                  scope="Development mechanics only; no training, acquisition, planning-quality, continual-efficiency or RSI result.",
                  runtime=dict(torch=torch.__version__, device="cpu", threads=torch.get_num_threads()),
                  seconds=time.perf_counter()-start,
                  sources={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ("action_e2e.py", "preflight_action_e2e.py", "e2e_core.py", "preflight_e2e_core.py")})
    Path(out).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
