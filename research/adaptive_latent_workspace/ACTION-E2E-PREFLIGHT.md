# Action/outcome interface: mechanics checked, agent untrained

E35-RESULT.md is the latest falsification: its short-sample archive selection
damages continual prediction. The action interface therefore starts from active
E2E state; it does not assume the failed archive is an improvement.

`action_e2e.py` reads a categorical observation and an executed action as two
disjoint token types. Only the subsequent observation is supervised, at the
action-token position, after it arrives. The existing immutable E2E core retains
KV and fast weights across real steps. Full outer differentiation traverses the
fast updates. Action selection, goal representation and useful planning are not
implemented by this adapter, and predictive meta-training is not a task-reward
objective. The four-state hidden-port fixture is solely a mechanics check.

`ActionState` and `ActionForecast` carry real observations and executed actions.
`ImaginedState` and `ImaginedForecast` branch without an update. Imaginary
forecasts cannot be passed directly to observation learning or persisted as
real state. These type checks prevent accidental misuse in this code; a caller
can still lie about an external observation. They are not an authenticity or
security guarantee. Callers also own real-time sequencing of their receipts.

All branches report forward tokens and attention score elements, including
branches the planner discards. Fast updates report gradient tokens and update
count. These counters are not total FLOPs or peak memory. Fingerprinting and
serialization have additional costs to charge in any benchmark. The model
fingerprint rejects incompatible slow weights or action/state encodings;
it does not implement migration after slow learning.

`preflight_action_e2e.py` completed in.4277s on one CPU thread (float64).
The frozen full-sequence dense reference differs by at most2.23e-15 in logits.
Twelve actual-action receipts differ from deliberately conflicting advisory
actions and match the fixture's executed outcomes. Changing a future outcome
changes the learned state without changing the earlier prediction or cache.
Five abandoned model branches consume10 forward tokens and zero updates;
subsequent real continuation remains exact. Actual save/load after five of12
adaptive actions reconstructs all later logits, receipts, work and final state.

Across four actions with two adaptive layers, including fast-dependent KV,
full second-order directional gradients agree with central differences within
1.78e-10 without clipping and2.75e-11 with active clipping. First-order surrogate
gradients differ. This tests the predictive objective; there is no differentiable
planner, actual reward optimization or trained policy here. All base parameters
remain unchanged during the inner loop.

The fixture's final persistent fast/KV tensors occupy12,288 bytes, excluding the
base, observation/counters, temporary graphs and Python overhead. This is not a
trained-task storage or efficiency result. Evidence and source hashes are in
`results/action_e2e_preflight.json`. No acquisition, retention, novel architecture,
long-horizon competence or recursive self-improvement is established by it.
