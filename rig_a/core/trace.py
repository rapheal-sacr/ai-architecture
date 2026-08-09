"""An inert-arm detector that runs before any metric is computed.

FOUR INSTANCES OF ONE DEFECT, and inspection found all four:

    B7   a control registered on a quantity no arm ever exercised -- no arm had
         free rank, so the arms never differed on the manipulated quantity
    B8   A4's ontology shift mutated `ontology_shift`, but the usage draw never
         calls `strata()`, so the mutation could not reach the measured quantity
    B20  A3 passed one `policy` to both compile calls, so nothing differed
         between them at all
    A4-usage  retained deliberately as an asserted-inert control

Output-level manipulation checks caught some of these and only after a number had
been published. A CALL-LEVEL check catches all four, statically, before a metric
exists -- because the question is not "did the number move" but "could it have".

TWO QUESTIONS, AND THEY ARE DIFFERENT FAILURES:

    did anything change?      B20's form. Snapshot the state before the treatment
                              and after it. Nothing changed => the arm is inert by
                              construction, whatever it is named.

    could the change be read? B8's form. Record every attribute the object reads
                              during the RECOMPILE phase. If the mutated set and
                              the read set are disjoint, the mutation cannot reach
                              the measurement, and a null there is definitional.

The second is the one worth having. A mutation that happens and is never read is
indistinguishable from no mutation at the output, and it is exactly what an
output-level check cannot see.

USAGE

    tr = ArmTrace(world, state=("provenance", "alive", "ontology_shift"))
    before = world.compile(...)
    tr.snapshot()
    ...treatments...
    with tr.recording():
        after = world.compile(...)
    tr.verdict()        # -> {"mutated": {...}, "read": {...}, "inert": bool}

`inert` is True when the arm could not have moved its own measurement. That is a
statement about the code, not about the run, so it holds for every seed at once.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import numpy as np


def _hook(cls):
    """Install a read-recording __getattribute__ once per class."""
    if getattr(cls, "_arm_trace_hooked", False):
        return
    orig = cls.__getattribute__

    def hooked(self, name):
        if not name.startswith("_"):
            try:
                log = orig(self, "_arm_trace")
            except AttributeError:
                log = None
            if log is not None and log["on"]:
                log["reads"].add(name)
        return orig(self, name)

    cls.__getattribute__ = hooked
    cls._arm_trace_hooked = True


def _freeze(v):
    """A comparable snapshot of a value, without assuming its type."""
    if isinstance(v, np.ndarray):
        return v.tobytes()
    if isinstance(v, (set, frozenset)):
        return frozenset(v)
    return v


@dataclass
class ArmTrace:
    """Records what an arm mutated and what its recompile actually read."""

    obj: object
    state: tuple
    _before: dict = field(default_factory=dict)
    _mutated: set = field(default_factory=set)

    def __post_init__(self) -> None:
        _hook(type(self.obj))
        object.__getattribute__(self.obj, "__dict__")["_arm_trace"] = {
            "on": False, "reads": set()}

    def snapshot(self) -> None:
        """Freeze the watched state. Call immediately before the treatment."""
        self._before = {k: _freeze(getattr(self.obj, k)) for k in self.state}

    @contextlib.contextmanager
    def recording(self):
        """Record attribute reads for the duration -- wrap the RECOMPILE only."""
        self._mutated = {k for k in self.state
                         if _freeze(getattr(self.obj, k)) != self._before.get(k)}
        log = object.__getattribute__(self.obj, "__dict__")["_arm_trace"]
        log["on"], log["reads"] = True, set()
        try:
            yield self
        finally:
            log["on"] = False

    def verdict(self, compile_args=None, recompile_args=None) -> dict:
        """`*_args` are the arguments the two compiles received.

        State is not the only thing an arm can vary. A3-rebuilt varies the DRAW
        POLICY ARGUMENT and mutates nothing, and a state-only detector calls that
        inert -- a false positive that would have condemned a valid arm. B20's
        form is precisely `compile_args == recompile_args AND nothing mutated`,
        so both halves have to be here or the check answers a different question
        than the one it was built for.
        """
        log = object.__getattribute__(self.obj, "__dict__")["_arm_trace"]
        read = set(log["reads"])
        reached = self._mutated & read
        args_differ = (compile_args is not None
                       and tuple(compile_args) != tuple(recompile_args or ()))
        if args_differ:
            reason = ""
        elif not self._mutated:
            reason = "nothing was mutated and both compiles took the same arguments"
        elif not reached:
            reason = "mutated but never read on the recompile path"
        else:
            reason = ""
        return {
            "mutated": sorted(self._mutated),
            "reached": sorted(reached),
            "unreachable": sorted(self._mutated - read),
            "args_differ": bool(args_differ),
            # inert unless SOMETHING the measurement can see differs: either a
            # mutation that gets read (B8's miss) or a differing argument (B20's)
            "inert": not reached and not args_differ,
            "reason": reason,
        }
