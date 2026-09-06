# E10 checkpoint-restoration correction

The first scored launch terminated before its first optimizer update. PyTorch's
optimizer loader left the `step` state on CPU; the pinned AdamGnT implementation
uses a per-parameter step tensor in CUDA arithmetic. The CPU preflight could not
expose that device mismatch. No completed arm result was produced.

The failure and original source/protocol hashes are preserved in
`results/e10_launch_failure.json` and the WD `e10/launch_failure.json`. It is a
runtime-adapter failure, not evidence that any of the four interventions fails.
Its exact startup wall time was not retained; it must not be reported as zero.

The adapter now moves every loaded optimizer-state tensor to its parameter's
device. This changes no hyperparameters, model equations, checkpoint values or
evaluation protocol. A CUDA preflight exercises the restored optimizer and the
critic/actor gradient-path checks before restarting scored execution in the
separate WD `e10-v2` directory. E8 files and results remain unchanged.
