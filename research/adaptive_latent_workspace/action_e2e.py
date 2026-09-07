"""Action-conditioned E2E interface; no trained agent or task result.

State symbols occupy [0, states); action symbols occupy [states, states+actions).
Read [current observation, executed action], predict only its subsequent real
observation, and update after it arrives. This is a categorical observation
adapter, not a natural-language or partially observed belief-state solution.

The separate real/imagined types catch accidental self-labeling by callers.
They are not a security boundary or proof that an external observation is true.
Core operations remain functional; callers own execution order and receipts.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import torch
import torch.nn.functional as F

from e2e_core import E2ECore, StreamState


@dataclass
class ActionState:
    stream: StreamState
    observation: int
    real_steps: int

    def detached(self):
        return ActionState(self.stream.detached(), self.observation, self.real_steps)


@dataclass
class ActionForecast:
    origin: ActionState
    action: int
    logits: torch.Tensor
    next_stream: StreamState
    work: dict


@dataclass
class ImaginedState:
    stream: StreamState
    observation: int
    depth: int


@dataclass
class ImaginedForecast:
    action: int
    logits: torch.Tensor
    branch: ImaginedState
    work: dict


class ActionE2E:
    version = "categorical-action-e2e-v1"

    def __init__(self, core: E2ECore, *, states: int, actions: int):
        if states < 2 or actions < 1:
            raise ValueError("Need at least two states and one action")
        if core.cfg.vocab != states + actions or core.cfg.chunk < 2:
            raise ValueError("Core needs disjoint state/action symbols and chunk >= 2")
        self.core, self.states, self.actions = core, states, actions

    def _state_symbol(self, value):
        if type(value) is not int or not 0 <= value < self.states:
            raise ValueError("Invalid observation symbol")

    def _tokens(self, observation, action):
        self._state_symbol(observation)
        if type(action) is not int or not 0 <= action < self.actions:
            raise ValueError("Invalid action symbol")
        return torch.tensor([observation, self.states + action], dtype=torch.long,
                            device=self.core.p["embed"].device)

    def initial_state(self, observation, *, params=None):
        self._state_symbol(observation)
        return ActionState(self.core.initial_state(params), observation, 0)

    def forecast(self, state: ActionState, action: int, *, params=None):
        if type(state) is not ActionState:
            raise TypeError("Real forecast requires an ActionState")
        tokens = self._tokens(state.observation, action)
        logits, following, elements = self.core.chunk_logits(tokens, state.stream, params)
        # The state-token position has no supervised target here. In particular,
        # we do not fit an advisory action label or predict our chosen action.
        return ActionForecast(state, action, logits[-1, :self.states], following,
                              dict(forward_tokens=2, attention_score_elements=elements,
                                   forwards=1, inner_updates=0, gradient_tokens=0))

    def observe(self, forecast: ActionForecast, outcome: int, *, mode="second_order"):
        if type(forecast) is not ActionForecast:
            raise TypeError("Only a real action forecast can receive observed labels")
        if mode not in ("second_order", "first_order", "frozen"):
            raise ValueError("Unknown update mode")
        self._state_symbol(outcome)
        target = torch.tensor([outcome], device=forecast.logits.device)
        loss = F.cross_entropy(forecast.logits[None], target)
        fast = forecast.origin.stream.fast
        norm = loss.new_zeros(())
        if mode != "frozen":
            gradients = torch.autograd.grad(loss, tuple(fast.values()),
                                            create_graph=mode == "second_order",
                                            retain_graph=True)
            norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
            scale = (self.core.cfg.clip / norm.clamp_min(1e-30)).clamp(max=1.0)
            updates = [scale * g for g in gradients]
            if mode == "first_order":
                updates = [g.detach() for g in updates]
            fast = {name: value - self.core.cfg.inner_lr * delta
                    for (name, value), delta in zip(fast.items(), updates)}
        # The newly arrived outcome is the next input observation, not a token
        # secretly inserted into the already-scored prefix. KV is retained.
        state = ActionState(StreamState(fast, forecast.next_stream.cache,
                                       forecast.next_stream.position),
                            outcome, forecast.origin.real_steps + 1)
        work = {**forecast.work, "inner_updates": int(mode != "frozen"),
                "gradient_tokens": 2 * int(mode != "frozen"),
                "inner_grad_norm": float(norm.detach())}
        receipt = dict(step=state.real_steps, observation=forecast.origin.observation,
                       executed_action=forecast.action, outcome=outcome,
                       provenance="external_observation")
        return loss, state, work, receipt

    def imagine(self, state: ActionState | ImaginedState, action: int):
        """Read-only greedy branch; it never invokes an inner update.

        A planner must sum work for every called branch, including discarded ones.
        This is a one-step model interface, not an implemented planning algorithm.
        """
        if type(state) not in (ActionState, ImaginedState):
            raise TypeError("Expected real state or an imagined branch")
        tokens = self._tokens(state.observation, action)
        with torch.no_grad():
            logits, following, elements = self.core.chunk_logits(tokens, state.stream)
            logits = logits[-1, :self.states]
            branch = ImaginedState(following, int(logits.argmax()),
                                   1 if type(state) is ActionState else state.depth + 1)
        return ImaginedForecast(action, logits, branch,
                                dict(forward_tokens=2, attention_score_elements=elements,
                                     forwards=1, inner_updates=0, gradient_tokens=0))

    def fingerprint(self):
        digest = hashlib.sha256(json.dumps(dict(version=self.version,
                                                states=self.states, actions=self.actions,
                                                config=asdict(self.core.cfg)),
                                           sort_keys=True).encode())
        for name, tensor in sorted(self.core.state_dict().items()):
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(value.dtype).encode())
            digest.update(str(tuple(value.shape)).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def payload(self, state: ActionState):
        if type(state) is not ActionState:
            raise TypeError("Only real state is persistent learner state")
        return dict(version=self.version, model_sha256=self.fingerprint(),
                    observation=state.observation, real_steps=state.real_steps,
                    stream=state.stream.payload())

    def restore(self, payload):
        if payload["version"] != self.version or payload["model_sha256"] != self.fingerprint():
            raise ValueError("Incompatible core or action encoding")
        self._state_symbol(payload["observation"])
        return ActionState(StreamState.restore(payload["stream"]),
                           payload["observation"], payload["real_steps"])
