# E36 shortcut: goal completion without learning

This supplemental control was added during E36 training, before scored action
results were available. It changes none of the frozen sources, training data,
selection rules or primary controls. Both constants are retained without choosing
the better action after seeing a world.

Each action independently follows a Hamiltonian cycle through all6 states.
Repeating action0 ignores observations, goals and memory but visits every state.
After each arrival, the new goal is uniform over the other5 states. Its distance
under that constant action is uniform1..5, with mean3 actions and asymptotic
completion rate1/3. The finite renewal recurrence gives170.444 expected goals
in512 stationary actions:

    R(0) = 0
    R(t) = (1/5) * sum_{d=1..min(5,t)} [1 + R(t-d)]

Direct execution on every frozen world gives:

| Regime | Action0: world0 / world1 | Action1: world0 / world1 |
|---|---:|---:|
| Stationary | 175 / 166 | 169 / 169 |
| Recurring | 176 / 175 | 174 / 170 |
| Drifting | 169 / 167 | 168 / 168 |
| Noisy | 130 / 139 | 145 / 128 |

All8,192 receipts were independently reconstructed without NavigationWorld:
source/action/outcome, noise, reward and goal transitions match. The312 complete
origin/action/map traversals verify the cycle property on all26 frozen maps.
These policies have no learned state and make no transition queries or updates.
Their fixed action and the simulator still occupy state; this is not zero total
system memory. Two-world acting/receipt times are .0024–.0053s per condition,
including world construction but excluding the separate audit and file writing.
These timings are not isolated kernel latency.

This falsifies **raw goal count as sufficient evidence of acquired map knowledge**
in this family. Neural comparisons remain pending. Preserve frozen/adaptive
ablations and cheap memory controls, and add both constants to interpretation.
A successful neural policy must earn its benefit over these alternatives.

Private map probes similarly measure what the specified query interface can
express. An incorrect probe does not prove every useful distinction has been
erased from weights: context-dependent access can also fail. Acquisition,
useful access and learning-induced behavioral change remain separate. Query-
context interventions may be needed after primary results; none is scored yet.

A later family must require contingent choices, for example strongly connected
action-union graphs where neither action alone tours all states. Test periodic
and reactive shortcuts too. Do not change the ongoing frozen worlds; report the
limitation. Even an expanded finite family cannot establish general reasoning.

Compact evidence: `results/e36_constant_control.json`. Full receipts are in WD
`alw-runs/e36-constant-control/complete.json`, tied to E36 data SHA256
`3b13d16a22d0b80d4c52aeb332a75be958e7b94818434070262465c2d35f4d59`
and frozen commit `2d3eff32a79a1561ba76d98a81b6a7f10bc92c04`.
