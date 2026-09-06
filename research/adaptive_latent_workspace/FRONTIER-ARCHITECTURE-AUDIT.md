# New model references: useful mechanisms and their limits

Read FALSIFICATION.md first. This is source analysis and narrow mechanism
execution, not a reproduction of any frontier model's benchmark results. E35
continues from frozen4e7730f; this review does not change its scored protocol.

The gallery clone is rasbt/LLM-architecture-gallery at8c820c3, whose models.yml
contains the supplied architectures. Initial direct requests for five WebP
assets returned403 with and without www; the clone does not contain those
images and a guessed website repository was unavailable. The user subsequently
attached six full PNG diagrams, adding Muse Glimmer. **All six attached diagrams
have now been inspected**, alongside checkpoint/source details and original
rendered Kimi/DeepSeek paper figures. Exact user attachments were copied to WD;
results/user_architecture_diagrams.json records their paths and SHA256 hashes.
Repositories were cloned, not scraped as rendered directory pages.
Only paper assets were downloaded separately; the Nanbeige PDF matches its
cloned LFS pointer hash. No large model weights were downloaded or executed.

Exact repository commits, configuration/source hashes and executed function
locations are in results/frontier_source_probes.json. Source roots on WD:
`AI Architecture Research/repos`; PDF/text/render artifacts are in
`AI Architecture Research/alw-runs/frontier-sources-2026-09-06` (Kimi text is in
the parent alw-runs directory). The targeted review covers architecture,
recurrence/cache implementation, reported attention/loop ablations, and relevant
agent-training/evaluation sections; it is not a claim to have reproduced all
training infrastructure or independently verified published scores.

## What each source contributes

**Kimi K3.** The [pinned report](https://github.com/MoonshotAI/Kimi-K3/blob/3cb39dfd32e51c3328e2e4b4af21341247d06c43/k3_tech_report.pdf)
uses69 KDA and24 Gated MLA layers, with global attention last, Block Attention
Residuals and Stable LatentMoE. Section2 bounds log-decay below by−5 to keep
16-token reciprocal decay rescaling within BF16's exponent range. That is a
numerical/kernel benefit, not a bound on the lifetime of remembered facts.
Attention Residuals expose earlier depth representations; they do not make all
past trajectory facts available indefinitely. The stated2.5× scaling gain
combines architecture, training and data changes, so it cannot be assigned to
one component from this report alone. Section4.2.6's autonomous tasks evaluate
actual final environment state with separate diagnostic/hidden verifiers and
submission budgets. That is a useful evaluation precedent. Section5.3.1
offloads KDA states together with MLA prefixes into CPU memory; persistence
there has storage/transfer cost. We did not measure those costs locally.

**GLM-5.3-Flash.** The [checkpoint configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/690b705278a3a58e538fcb37c2ca8b5f9511213c/config.json)
specifies34 KDA and11 sparse MLA layers, four mHC streams and a separate MTP
layer. The supplied [2602.15763 paper](https://arxiv.org/pdf/2602.15763)
is *GLM-5: from Vibe Coding to Agentic Engineering*, not a5.3-Flash report.
Its GLM-9B ablation is still informative: after190B adaptation tokens, SimpleGDN
has67.03 RULER@128K versus75.28 for the full-attention baseline, while HELMET-ICL
improves. Task-average quality and exact retrieval need separate tests. Its
term continual-training refers here to converting a pretrained model, not our
unannounced lifelong acquisition/retention experiment. Empirical sparse-attention
parity also does not establish exact dense-attention equivalence for arbitrary
queries. The paper's slide-improvement pipeline uses externally designed rewards
and later renderer repairs for observed reward exploitation; it does not remove
the need to validate the evaluator. Do not attribute these results directly to
the different5.3-Flash checkpoint.

**DeepSeek V4-Pro.** The [released inference code](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/b5968e9190ef611bbf34a7229255be88a0e937c1/inference/model.py)
implements a128-token exact window, compressed sparse attention at ratio4 and
heavily compressed attention at ratio128. Among61 main layers,30 use ratio4
and31 ratio128; the extra ratio0 entry is MTP. Learned gated pooling, overlapping
ratio4 blocks and a separate indexer are concrete reusable mechanisms. But
Indexer.forward computes query scores against **every** available compressed
index key before top-k; HCA reads all heavily compressed entries. With growing
history, these branches still have growing storage/read work. Top-k bounds the
expensive final sparse read, not the entire retrieval pipeline. The
[paper](https://arxiv.org/pdf/2606.19348) reports retrieval degradation beyond128K
and treats online learning as future work. None of this negates its large
constant-factor improvements. We have not executed its CUDA/quantized kernels
or measured whether its learned pooling loses any particular real fact.

**Qwen3.8-27B.** Its [configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
actually names the Qwen3_5 implementation family. It confirms48 gated delta
layers and16 full-attention layers, partial RoPE and a separate MTP head. The
current Transformers clone atc93057d implements scalar per-head decay and
error-correcting delta writes; the recurrent state is FP32. Learned recurrent
updates are useful sequence compression, but they are distinct from E2E's
gradient-trained fast MLP and its training through future SGD steps.

**Nanbeige4.2-3B.** The [source/configuration](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/3384e426066d1a49c3aea90a7190b81260a6533f)
executes the same22-layer stack twice. The unchanged loop-cache helper yields
44 distinct cache indices. Its report says more passes give marginal gains,
slower training and less stable optimization, and that sharing caches lowers
quality. Training from scratch with the loop outperforms adding it later.
Those are reported ablations, not locally repeated measurements. The repository
also contains options intended for the forthcoming4.5 model (LoopSplit, mHC,
depth attention, n-gram embeddings); the4.2 configuration does not enable them.
Their presence in code cannot be credited to the released4.2 model. The two-pass
design supports weight reuse, not an inference-time guarantee that arbitrary
extra loops improve reasoning. Our E28 failures remain relevant.

**Muse Glimmer30B.** The added diagram and [checkpoint](https://huggingface.co/meta-models/Muse-Glimmer-30B/blob/a4e59da52a7bc87ae7251dd5545c0dd437c44b68/config.json)
agree on39 local and13 global layers,52 layers total,2048-token windows, width6656,
32 query/2 KV heads, and pre/post RMS normalization. Transformers source confirms
local-only RoPE, global NoPE, scaleless Q/K normalization and output gating.
This gives a useful alternative to delta recurrence: retain exact local history
and pay for periodic global access. It does not preserve arbitrarily old
independent facts in fixed space. Its model card describes separate DFlash
speculative decoding; verifying draft tokens against the target model accelerates
sampling but is not a verifier of real-world truth or a learning-procedure update.
No Muse weights, drafter, reported speedup or agent benchmarks were run locally.

## Local source probes and complete-memory objections

audit_frontier_sources.py executes unchanged PyTorch recurrence/helper bodies
from pinned source; only optional kernel-dispatch decorators are removed. A
constructed initial two-row state holds an old distinction in row0. Subsequent
writes address orthogonal row1. With scalar retention.99, only0.000033919 of the
old component remains after1024 writes; with channel-wise retention.999 for
row0 and.99 for row1,0.358976 remains. Both match their elementary exponential
predictions within the declared numeric tolerance. These are chosen gate values,
not matched trained-model comparisons or measured forgetting of acquired tasks.
They demonstrate what the interfaces permit: channel selectivity can help,
while a legal forget gate still provides no universal retention guarantee.

Logical batch1 main-decoder accounting, excluding weights, vision, MTP,
convolution buffers, allocators and other unspecified state:

| Model | Growing BF16 attention state | Additional recurrent/index state |
|---|---:|---:|
| Qwen3.8-27B |64 KiB/token |144 MiB fixed FP32 recurrent matrices |
| GLM-5.3-Flash |11 KiB/token MLA latents |136 MiB fixed FP32 recurrent matrices; sparse index state additional |
| Nanbeige4.2-3B |176 KiB/token across44 execution caches |Other buffers additional |
| DeepSeek V4-Pro |7,928 B/token asymptotic compressed main KV |1,920 B/token extra BF16 index keys; local/pending buffers additional |
| Muse Glimmer30B |13 KiB/token in13 global layers |81,748,992 B bounded local past KV with2047 entries per local layer; current chunk additional |

These are derived storage counts, not runtime peaks or claims about deployed
quantization. DeepSeek's reference comments explicitly distinguish intended
quantization from BF16 cache simulation. The gallery's7.7 KiB estimate matches
main compressed KV alone; adding the source's indexer increases that subtotal
to9,848 B/token. Smaller per-token cache slopes do not establish bounded total
memory, and fixed recurrent state does not make the remaining global attention
cache constant. CPU/GPU/disk copies also count when evaluating a real system.

Muse's gallery52 KiB/token value is the naive all52-layers-unbounded subtotal.
The current model constructs DynamicCache with its local/global configuration,
and DynamicSlidingWindowLayer retains only window−1 past entries. Its persistent
KV growth is therefore13 KiB/token plus roughly78 MiB local storage once windows
fill, rather than52 KiB/token indefinitely. This is source-derived accounting,
not a measured process peak; full-prefill intermediates and backend choices matter.

## Consequences for the candidate, mapped to the three roots

| Candidate change | Root attacked | Required falsification |
|---|---|---|
| Use a learned recurrent stream state for recent context, plus selective access to retained evidence | Retention and useful computation | Equal total state including recurrent matrices/indexes; rare exact queries; evidence retrieval cost as history grows |
| Put gradient-adapted task state behind a representation boundary and preserve validated alternatives | Retention/interference | E35 unannounced selection/capacity tests; encoder revisions must pay compatibility/refresh cost; final-suffix restriction must remain explicit |
| Reuse reasoning layers with input/evidence reinjection and learned compute allocation | Useful computation | Train for the exercised depths; distinct caches where representations differ; compare success at equal total work; extra loops may hurt |
| Validate learner/procedure changes on later real outcomes under a fixed evaluator | Identifying evidence and useful computation | Held-out events, trial/rejection costs, evaluator exploitation and equal-budget ordinary learning controls |

The most useful transferable idea is **distinct state lifetimes with explicit
read/write costs**, not a universal3:1 layer recipe. Recurrent hidden state,
per-token evidence, adapted weights, reasoning-pass caches and learning-procedure
state encode different things and need different retention tests. Mixing these
existing components is not established novelty. No supplied leaderboard ordering
isolates architecture from scale, data, training, harness and inference budget.

The next full-system step after the archive assay must reconnect E2E's learned
state to actual actions, learned transition predictions and bounded reasoning.
Imagined rollouts must not become real-observation training targets. A preserved
grammar predictor alone cannot establish that the agent acquires useful behavior,
compresses general experience or recursively improves its learning procedure.
All of those requirements remain open.
