# Fixed-point source audit: a possible stability mechanism, not a correctness test

Cloned `https://github.com/locuslab/deq` on WD, pinned at
`1fb7059d6d89bb26d16da80ab9489dcc73fc5472`. Read the top-level README,
`lib/jacobian.py`, the Broyden/Anderson implementations in `lib/solvers.py`, and
the sequence transformer's recurrent/equilibrium forward and backward-hook path.
Hashes are retained in `results/deq_source_audit.json`. No training, dataset
installation or published numerical score was reproduced.

The sequence model solves for a fixed point without retaining an unrolled
forward autograd graph, then uses another solve involving Jacobian-vector
products for implicit differentiation. Its Jacobian regularizer estimates a
shape-normalized squared Frobenius norm using random vectors. Spectral-radius
estimation is an optional separate evaluation path. This is established prior
art for stable implicit recurrent computation, not an invention of this project.

Several implementation details matter before borrowing the mechanism:

- The solvers return the best residual state. Returned `nstep` identifies the
  iteration attaining the lowest residual; it is not necessarily the total
  number of function evaluations. Broyden may add line-search evaluations.
- Trace arrays may be padded after stopping. Their length does not by itself
  establish actual work. A port must count function calls directly.
- Broyden allocates U/V histories proportional to the iteration threshold and
  state size. Anderson allocates m copies of state/function values plus a small
  linear system. Avoid equating removal of the unrolled autograd graph with
  zero solver workspace or zero backward computation.
- Residual norms in the inspected solvers aggregate a batch. They are not an
  independent semantic correctness check for every sample.
- The Anderson implementation calls legacy `torch.solve`. A direct local probe
  on torch2.5.1 raises the documented removal error. Any future execution needs
  an explicit compatibility port, with argument order/return semantics checked;
  the upstream clone is unchanged.

A fixed point can be exactly wrong. For instance, the map f(z,x)=z/2 has a
unique attractive fixed point z=0 and zero residual there, regardless of whether
an external target is zero or one. Stability or solver convergence therefore
cannot replace final-task correctness or independent outcome feedback. This is
an elementary counterexample, not a trained-model experiment.

E28 motivates investigating stability because extra recurrence damages answers
on unchanged inputs and weights. The ongoing E29 test isolates varying training
depth at equal message work. This DEQ audit does not select a follow-up from
partial E29 scores. Implicit solving, Jacobian penalties and explicit consistency
losses remain different untested interventions. A learned equilibrium would
still need acquisition, old-task retention, size transfer and full cost tests.
