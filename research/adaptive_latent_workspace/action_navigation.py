"""Unknown-port navigation fixtures and an observation-only planner control.

Every action defines a fresh directed Hamiltonian cycle. State labels and legal
actions are public; destinations, hidden context changes and future goals are
not policy inputs. This finite simulator is not a general-reasoning benchmark.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import json
import random


def domain_seed(seed, domain, index=0):
    return int.from_bytes(hashlib.sha256(f"action-navigation|{seed}|{domain}|{index}".encode()).digest()[:8], "big")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def graph(seed, states, actions):
    rng = random.Random(seed)
    destinations = [[None]*actions for _ in range(states)]
    for action in range(actions):
        order = list(range(states))
        rng.shuffle(order)
        for source, target in zip(order, order[1:]+order[:1]):
            destinations[source][action] = target
    return destinations


@dataclass(frozen=True)
class NavigationObservation:
    state: int
    goal: int


class NavigationWorld:
    """Agent receives observe()/step() observations, never this object's maps."""
    def __init__(self, maps, durations, slots, *, goal_seed, initial_state=0,
                 noise_rate=0., noise_seed=0):
        assert maps and len(durations) == len(slots) and all(d > 0 for d in durations)
        self._maps = maps
        self.states, self.actions = len(maps[0]), len(maps[0][0])
        assert all(len(m) == self.states and all(len(row) == self.actions for row in m) for m in maps)
        assert all(0 <= x < self.states for m in maps for row in m for x in row)
        assert all(0 <= s < len(maps) for s in slots)
        self._durations, self._slots = list(durations), list(slots)
        self._schedule = [slot for slot, duration in zip(slots, durations) for _ in range(duration)]
        self._rng = random.Random(goal_seed)
        assert 0 <= noise_rate <= 1
        self._noise_rate = noise_rate
        self._noise_rng = random.Random(noise_seed)
        self._state, self._step, self._completed = initial_state, 0, 0
        self._goal = self._next_goal(initial_state)

    def _next_goal(self, previous):
        return self._rng.choice([s for s in range(self.states) if s != previous])

    def observe(self):
        return NavigationObservation(self._state, self._goal)

    def step(self, action):
        if type(action) is not int or not 0 <= action < self.actions:
            raise ValueError("Invalid action")
        if self._step >= len(self._schedule):
            raise RuntimeError("Action budget exhausted")
        old = self.observe()
        slot = self._schedule[self._step]
        self._state = self._maps[slot][self._state][action]
        if self._noise_rate and self._noise_rng.random() < self._noise_rate:
            self._state = self._noise_rng.randrange(self.states)
        self._step += 1
        reward = int(self._state == self._goal)
        if reward:
            self._completed += 1
            self._goal = self._next_goal(self._goal)
        # The receipt excludes private map/context information. Reward refers
        # to the OLD goal; the next observation may already request a new goal.
        receipt = dict(step=self._step, observation=old.state, goal=old.goal,
                       executed_action=action, outcome=self._state, reward=reward,
                       next_goal=self._goal)
        return self.observe(), reward, receipt

    def payload(self):
        return dict(maps=self._maps, durations=self._durations, slots=self._slots,
                    rng=self._rng.getstate(), state=self._state, step=self._step,
                    completed=self._completed, goal=self._goal,
                    noise_rate=self._noise_rate, noise_rng=self._noise_rng.getstate())

    @classmethod
    def restore(cls, payload):
        world = cls(payload["maps"], payload["durations"], payload["slots"], goal_seed=0)
        world._rng.setstate(payload["rng"])
        world._state, world._step = payload["state"], payload["step"]
        world._completed, world._goal = payload["completed"], payload["goal"]
        world._noise_rate = payload.get("noise_rate",0.)
        if "noise_rng" in payload:
            world._noise_rng.setstate(payload["noise_rng"])
        return world


class ObservedTransitionPlanner:
    """BFS through observed directed edges, then nearest unknown action.

    Unknown edges are not guessed to lead to the goal. Reobserved edges replace
    older destinations without a context label. No inverse edges are inferred.
    Goal-directed actions take priority over exploration; ties are deterministic.
    """
    def __init__(self, states, actions, *, capacity=None):
        self.states, self.actions, self.capacity = states, actions, capacity
        self.records = OrderedDict()
        self.writes, self.evictions, self.edge_visits = 0, 0, 0

    def observe_transition(self, before, action, outcome):
        key = (before, action)
        self.records.pop(key, None)
        self.records[key] = outcome
        self.writes += 1
        if self.capacity is not None:
            while len(self.records) > self.capacity:
                self.records.popitem(last=False)
                self.evictions += 1

    def act(self, observation):
        queue = deque([(observation.state, None, 0)])
        visited = {observation.state}
        frontier = []
        while queue:
            here, first, distance = queue.popleft()
            if here == observation.goal and first is not None:
                return first
            for action in range(self.actions):
                self.edge_visits += 1
                successor = self.records.get((here, action))
                if successor is None:
                    frontier.append((distance, here, action, action if first is None else first))
                elif successor not in visited:
                    visited.add(successor)
                    queue.append((successor, action if first is None else first, distance+1))
        if frontier:
            return min(frontier)[-1]
        # A complete observed map can be stale/disconnected after hidden changes.
        # Executing an action collects a fresh transition; no hidden reset occurs.
        return self.writes % self.actions

    def payload(self):
        return dict(states=self.states, actions=self.actions, capacity=self.capacity,
                    records=[(s,a,y) for (s,a),y in self.records.items()], writes=self.writes,
                    evictions=self.evictions, edge_visits=self.edge_visits)

    @classmethod
    def restore(cls, payload):
        result = cls(payload["states"], payload["actions"], capacity=payload["capacity"])
        result.records = OrderedDict(((s,a),y) for s,a,y in payload["records"])
        result.writes, result.evictions, result.edge_visits = payload["writes"], payload["evictions"], payload["edge_visits"]
        return result
