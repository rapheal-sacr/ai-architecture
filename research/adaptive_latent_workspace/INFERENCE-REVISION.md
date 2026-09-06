# Learn evidence allocation before expanding the memory substrate

Read `FALSIFICATION.md` first. E19–E21 change the next architecture decision.
Procedure self-application changes a plasticity/retention tradeoff, not a
uniformly improving algorithm. Correct context helps much more than width in
E20. E21 then repairs much of that failure without oracle information, extra
student parameters or a larger buffer, but causes negative input-shift transfer.

E22 implemented and falsified the proposed causal gate as a sufficient repair.
It learns from recorded prediction errors but loses to fixed fast/prior on all
conflicting, independent and alternating worlds, and fixed slow/prior on all
input-shift worlds, at 1.95 times fast/prior execution cost. Coupled changes give
a small scoped gain. The original design below is retained to show what was
intended; `protocol_e22.json` specifies the actual implementation, which uses
prior assignment for the learned gate and includes a fast/posterior comparator.

## State and causal update

Keep fast and slow summaries of observed input/output relationships, the shared
conditional learner, bounded contextual replay and a small persistent gate.
Before the next outputs arrive, the gate uses only the summaries, current inputs
and lagged prediction diagnostics to form a context for prediction. It cannot
use the current output, true regime, task family or a boundary indicator.

After the outputs arrive, that prediction supplies an actual error signal for
the gate. Train the gate against subsequent predictive usefulness rather than
summary reconstruction or raw surprise. Update the fast/slow evidence states.
Assign current observations to memory only after deciding which newly available
evidence should change their context. Prediction and memory assignment are
different causal moments; using a new outcome to annotate its stored experience
does not authorize using it in the earlier prediction.

Start with a single shared model and a small gate, without adding a hypernetwork,
more modules or another expensive planning loop. Compare fixed slow, fixed fast
and fixed-mixture controls with the same learner and charged storage. Any extra
model forwards, backpropagation, gate inference and adaptation state count.
E21's inexpensive fast/posterior control must remain a strong comparator.

## Falsification that must accompany the implementation

The current two families have a shortcut: input means change in the input-shift
family while conflicting functions share a stable input distribution. A gate
could exploit that family correlation instead of learning when the conditional
relationship changed. New tests must couple and decouple input-distribution
changes from function changes. Include a stream where the two change processes
have independent clocks and neither family nor boundary is supplied.

Report causal prequential error, all censored recovery failures and complete
operation/storage counts. Require the gate to improve the tradeoff against both
fixed timescales; a win on only one easy family repeats E21's failure. Check that
training-label contributions to context do not become a prediction shortcut.
If gate improvement is claimed, hold student and experience state fixed while
keeping/reverting the gate, as E19 did for the updater.

Only after evidence allocation earns that claim should it be integrated with
protected modules, learned dynamics and procedure self-application. Each still
has a separate failure: module recruitment confuses novelty with underfit;
planning can fail with accurate dynamics; procedure adaptation can specialize;
finite memory checks do not preserve arbitrary rare facts.

## Root and novelty boundaries

The gate attacks identifying evidence and the assignment of experience to
retained conditional models. Its computational advantage must be measured,
not assumed. It cannot reveal a new independent hidden regime before evidence
arrives or compress arbitrary independent facts without loss.

TTT-E2E and Titans motivate training adaptive neural state for later usefulness;
their supplied papers do not establish exact persistent recall. LEO already
learns latent adaptation, and HyperCL already generates and protects weights
from task embeddings. Learned filtering, multi-timescale state and gating are
also established ideas. This proposed component is an evidence-driven research
implementation, not an established novelty claim. The distinctive full
architecture, semantic compression, general reasoning and efficient autonomous
long-horizon behavior remain open.

## After E22: evidence geometry is the next causal intervention

Raw cross-moments E[[x,1] y] depend on the input distribution even when the
conditional function is unchanged. A regularized linear coefficient estimate
(E[[x,1][x,1]^T] + ridge I)^-1 E[[x,1]y] separates these effects exactly only in
a noiseless full-rank linear setting with zero ridge. With nonlinearity it is a
distribution-weighted projection; it can still change under input shift. This
limitation is a falsification target, not a claim of invariant task identity.

Test the coefficient descriptor against the same raw descriptor, prediction
and post-outcome assignment timing, decay choices, student size, observations
and reservoir. Count covariance storage and all matrix solves. A linear
invariance preflight validates the algebra; fresh nonlinear independent and
alternating streams determine whether it helps the actual problem. No outcome
of that intervention alone establishes learned semantic compression, reliable
self-improvement or general reasoning. The wider architecture remains open.
