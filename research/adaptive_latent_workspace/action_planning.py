"""Differentiable finite-horizon planning from E2E transition predictions.

The Bellman search is fixed code, not a learned reasoning algorithm. Hypothetical
queries share the CURRENT real memory/history; they do not update fast weights
or simulate future learning. This is a Markov-model approximation under the
current belief, not an exact partially observed Bayes-adaptive planner.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PlanningConfig:
    horizon: int = 8
    discount: float = .95
    temperature: float = .1
    exploration: float = .1

    def validate(self):
        assert self.horizon >= 1 and 0 < self.discount <= 1
        assert self.temperature > 0 and 0 <= self.exploration <= 1


def transition_logits(adapter, real_state, *, params=None, vectorized=True):
    """Independent read-only hypothetical pairs; all queries remain charged.

    Vmap batches independent core calls, not a fake sequence of observations.
    Discarded cache outputs are never installed into the real learner state.
    """
    model = adapter.core
    device = model.p["embed"].device
    pairs = torch.tensor([(s, adapter.states+a) for s in range(adapter.states)
                          for a in range(adapter.actions)], device=device, dtype=torch.long)
    def query(tokens):
        return model.chunk_logits(tokens, real_state.stream, params)[0][-1, :adapter.states]
    scores = torch.vmap(query)(pairs) if vectorized else torch.stack([query(pair) for pair in pairs])
    elements = sum(model.cfg.heads*2*(len(k)+2) for k,v in real_state.stream.cache)
    n = len(pairs)
    work = dict(hypothetical_queries=n, forward_tokens=2*n,
                attention_score_elements=elements*n, inner_updates=0,
                gradient_tokens=0, vectorized_batches=int(vectorized))
    return scores.reshape(adapter.states, adapter.actions, adapter.states), work


def goal_values(probabilities, goal, config):
    """Discounted probability of hitting the current goal within H transitions.

    Reaching the goal terminates this plan. The actual environment then supplies
    a new goal; no future goal sequence or map enters these computations.
    """
    config.validate()
    n, actions, outcomes = probabilities.shape
    assert n == outcomes and 0 <= goal < n
    hit = torch.nn.functional.one_hot(torch.tensor(goal, device=probabilities.device), n).to(probabilities.dtype)
    value = torch.zeros_like(hit)
    for _ in range(config.horizon):
        q = probabilities @ (hit + config.discount * (1-hit) * value)
        value = q.max(-1).values
    return q


def choose_distribution(adapter, state, goal, config, *, params=None, vectorized=True):
    scores, work = transition_logits(adapter, state, params=params, vectorized=vectorized)
    probabilities = scores.softmax(-1)
    q = goal_values(probabilities, goal, config)
    logits = q[state.observation] / config.temperature
    policy = (1-config.exploration) * logits.softmax(-1) + config.exploration/adapter.actions
    work.update(bellman_iterations=config.horizon,
                bellman_transition_terms=config.horizon*adapter.states**2*adapter.actions)
    return policy, work


def sample_action(probabilities, uniform):
    assert 0 <= uniform < 1
    cumulative = probabilities.detach().double().cumsum(-1)
    return min(int((cumulative <= uniform).sum()), len(probabilities)-1)
