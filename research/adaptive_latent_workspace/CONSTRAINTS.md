# Constraints before further architecture claims

The current design is not a demonstrated general continual learner. E4 is an
external counterexample to its efficient-transfer claim. The following arguments
clarify which failures can be repaired architecturally and which require an
assumption about the task stream. These are elementary derivations, not claims
of new theorems or results measured by E1–E6.

## Three roots

1. **Insufficient identifying evidence.** Two currently possible environments can
   require different actions or predictions while presenting the same history.
   Neither better memory nor more computation alone resolves that ambiguity.
2. **Loss or interference in retained information.** Experience may have supplied
   the distinction, but state compression, shared updates or capacity allocation
   can make it unavailable later. Preserving everything indefinitely is not a
   bounded-memory solution.
3. **Cost of turning retained information into useful behavior.** Routing,
   adaptation, planning and checking consume time. A retained model that cannot
   be selected or composed within the useful task lifetime does not provide its
   nominal benefit. Wrong model-based plans can make future evidence worse.

Representational underfit and loss of plasticity can affect roots 2 and 3.
Prediction error alone does not identify the root: E4's changing input
distribution and E1's conflicting conditional functions can both cause surprise.

| Proposal | Root attacked | Current evidence or failure |
|---|---|---|
| Infer context from observation/outcome history | 1 | E2 reuses four hidden regimes; oracle identification is still better |
| Protect established modules during temporary adaptation | 2 | E5's overwrite ablation removes most of the gain in the abrupt family |
| Check active model before searching alternatives | 3 | E5 reduces measured routing work; overlapping-context false acceptance remains open |
| Retain a bounded sample and test shared consolidation | 2 and 3 | E6 is an exploratory test; finite anchors cannot certify all rare knowledge |
| Renew low-utility neural features | 2 and 3 | E7 improves whole-stream error at extra cost; E8 does not prevent closed-loop forgetting |
| Plan using learned multi-step dynamics | 3 | E14 gives a costly scoped gain; E15/E16 expose remaining search limits even with accurate dynamics |

## An identification lower bound

Suppose a context Z is drawn uniformly from K alternatives immediately before a
query, independently of past history. The new input X also gives no information
about Z. Let its noiseless target be f_Z(X). Before seeing the new outcome, the
best squared-error prediction is the average function

    f_bar(x) = (1/K) sum_k f_k(x).

Its minimum conditional squared error, per output dimension, is

    (1/K) sum_k (f_k(x) - f_bar(x))^2.

Independent zero-mean output noise adds its variance. The derivation is the
identity E[(Y-a)^2] = Var(Y) + (E[Y]-a)^2. More memory cannot remove a new draw
that is independent of everything remembered. Observations after the change
can reduce uncertainty; charging the delay is essential.

This is **not a numeric bound asserted for E1**: E1's regular context schedule
could itself be learned and predicted. Random, unannounced changes are needed
to test the stronger identification claim. Current tests do not measure a
universal minimum number of adaptation samples.

## A bounded-memory counterexample

Let X contain N independent fair bits that will later be queried. Suppose the
complete retained state M has at most 2^B possible values, no external copy is
available, and a decoder reconstructs the bits with average error D <= 1/2.
Write h2 for binary entropy in bits. Then

    B >= I(X; M) = N - H(X | M) >= N * (1 - h2(D)).

For the final inequality, the decoder's error bit determines each original bit
given M; bound the conditional entropy by the sum of the error-bit entropies,
then use concavity of h2. Exact recovery therefore requires at least N bits.
At fixed B, maintaining small error for an indefinitely growing collection of
independent random facts is impossible. Compression can exploit structure,
restricted future queries or allowed loss; it cannot remove this requirement
for arbitrary independent facts.

The retained state includes weights, optimizer state, buffers, indices and every
other persistent representation. Our current tensor-byte accounting omits some
of those terms, so it cannot be substituted for B to claim a measured
information-theoretic optimum. A compressed student or summary must be tested on
rare queries and changed distributions, not only on its compression sample.

## What quantity should bind the experiments?

For a specified recurring task distribution, measure **amortized recovery cost
relative to useful context lifetime**. Recovery cost includes identification,
weight updates, replay, routing and consolidation. For a declared competence
threshold, record the distribution of observations/time to recovery, recurrence
frequency, useful duration and late retention loss. A system can improve a
whole-stream mean while repeatedly paying to relearn excluded contexts; E3 and
E4 show why the late-time quantities matter.

This is a candidate bottleneck to measure, not an established universal scalar
bound. E1–E16 retained early-window and late-window losses rather than a
thresholded recovery-time distribution; E17/E18 add the limited assay below. They establish interference and routing
effects, and expose capacity exhaustion; they do not yet establish which single
quantity binds a general autonomous agent's improvement rate.

Declaring “information”, “compute” or “verification” universally binding without
the environment and cost model would conceal that missing measurement. The
architecture must earn a narrower claim by crossing all three roots on a
specified task family, then survive transfer and long-horizon closed-loop tests.

## First censored recovery measurement

E17 implements a thresholded assay on unpredictable 16–64-batch recurring
segments. Both inherited self-applied procedures recover in 115/228 segments
under conflicting functions, compared with 227/228 under input shift. Every
non-recovery is retained as censoring at the segment's actual end. The threshold
is three consecutive clean normalized-MSE batches at or below 0.2 and is used
only for evaluation. See `E17-RESULT.md` for costs and scope.

This makes the proposed recovery-cost/lifetime ratio operational on one stream
family, but not a universal law. Identification and adaptation occupy a large
fraction of useful segment life in the conflicting case. The assay cannot
attribute all of that fraction to one irreducible root, and lower average error
does not prove improvement at fixed total work.
