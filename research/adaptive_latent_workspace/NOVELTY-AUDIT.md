# Novelty is not established

The prototype is a new implementation and experimental line relative to the
user's WAM and finite-program architectures. That does not establish research
novelty. Protected modules, rehearsal, soft targets, context gating, temporary
model copies and consolidation have substantial prior art.

The cloned [continual-learning repository](https://github.com/GMvandeVen/continual-learning)
at `e6d795aa81b9cef742b8de76cb71222d4d1ce00b` already provides separate networks,
context gating, experience and generative replay, functional regularization and
task-free streams. Source inspected: `README.md`, `train/train_stream.py`, and
`data/datastream.py`. These are source-code observations, not claims that every
method has been replicated here.

In particular, the stream trainer copies a previous model at periodic update
boundaries and supports replay with hard or soft targets. The datastream can
omit context labels with `return_context=False`; the presence of task-free
training therefore does not itself distinguish this proposal. Some other
configurations do pass context labels, so baselines must be selected and audited
per configuration rather than described collectively as oracle-assisted.

TTT/Titans supply adaptive neural-state ideas; Macaron supplies modular
isolation; Continual Backprop supplies feature renewal. Those mechanisms should
be credited rather than relabeled as inventions here. The present candidate's
specific combination of error-based admission, protected temporary adaptation,
active-first routing and sample-tested merging is an experiment, not a proven
novel architecture. A more comprehensive literature and implementation
comparison is still required before making a novelty claim.

Current contribution to this research project is the falsifiable implementation,
its controls, source audits and negative transfer results. The current evidence
does not support either broad architectural novelty or the user's overall
continuous-learning/long-horizon capability goal.

## Learned-model action search

The cloned [PETS / handful-of-trials](https://github.com/kchua/handful-of-trials) source at `77fd8802cc30b7683f0227c90527b5414c0df34c` already learns dynamics and plans with cross-entropy search. Inspected `dmbrl/controllers/MPC.py`, `dmbrl/misc/optimizers/cem.py`, `dmbrl/config/default.py` and `dmbrl/config/cartpole.py`. The implementation stores observed transition data, warm-starts shifted action sequences, supports deterministic/probabilistic ensembles and several trajectory-sampling schemes, and uses supplied state/action cost functions. The default trajectory-sampling setup uses five networks and twenty particles; the Cartpole configuration uses a much larger network/search budget than E14. These are source observations, not reproduced benchmark scores.

E14 is a small independent deterministic world model with bounded replay that also learns reward. Combining it with CEM or reusing previous action sequences is not novel. Its lack of ensemble uncertainty and its small search budget are material differences from this prior work. E15 specifically shows that even accurate predictions do not remove the observed search limitation at the existing budget.

## Policies as persistent search proposals

The cloned [POPLIN source](https://github.com/WilsonWangTHU/POPLIN/tree/edd8dba50f9049c6164eda774602bef0c299cb51) at `edd8dba50f9049c6164eda774602bef0c299cb51` is direct prior art for learning a policy to guide model-based search. Inspected the README, `dmbrl/misc/optimizers/POPLIN_A.py` training path, `POPLIN_P.py` proposal/trace path, `policy_network/BC_A_policy.py` and `dmbrl/config/gym_pendulum.py`. Action-space variants use a neural policy to initialize or guide action proposals; parameter-space variants search policy weights and retain elite weights, returns and start states. The action-policy training path constructs real state/action data and an imagined dataset and passes both to policy training. The displayed return-based filtering block in that path is commented out; do not describe that code as active return filtering.

Learning a reusable search procedure, distilling search into a policy, or searching a low-dimensional policy instead of each action is therefore not itself an invention of this project. E16's fixed temporal blocks also do not establish novelty. A potential distinction would need a precise new mechanism and evidence against such baselines, including the price of collecting and validating procedure-training data.

The cloned Pendulum configuration uses an external MBBL wrapper, a 1,000-step task horizon, a 30-step planning horizon, population 500, 50 elites and five iterations, with supplied state/action costs. That wrapper's semantics were not audited or installed; these are not comparable benchmark scores or a replication of E14's Gymnasium reward. No POPLIN execution or numerical result was reproduced here.

## Online procedure changes

E18 combines learned update multipliers, retrospective differentiation, persistent student state, replay and finite candidate trials. Each component has prior art in the already reviewed learned-optimization, TTT, modular/replay and empirical self-improvement sources. The particular independently written combination is not enough to establish scientific novelty. Its input-shift gain and conflicting-function failure are reproducible project results, not evidence that online meta-learning or copied candidate learners were invented here.

## Latent adaptation and compressed parameter memories

The cloned [LEO implementation](https://github.com/google-deepmind/leo/tree/de9a0c2a77dd7a42c1986b1eef18d184a86e294a) at `de9a0c2a77dd7a42c1986b1eef18d184a86e294a` already encodes examples into latent representations, optimizes those latents with inner-loop gradients, decodes classifier weights and optionally fine-tunes the decoded weights. Inspected README and `model.py`, including `leo_inner_loop`, `finetuning_inner_loop`, encoder, relation network and decoder. This is episodic labeled classification using supplied pretrained embeddings; it is not a demonstrated task-free lifelong inference mechanism. No training or published accuracy was reproduced here. Low-dimensional adaptive codes and weight decoders cannot by themselves be claimed as new.

The cloned [Continual Learning with Hypernetworks source](https://github.com/chrhenning/hypercl/tree/e32567889f772f8de783a437ff0beb2d426bc7b6) at `e32567889f772f8de783a437ff0beb2d426bc7b6` already generates task-conditioned model weights and regularizes generated weights for previous task embeddings. Inspected README, `utils/hnet_regularizer.py` target creation and fixed-output regularizer, and task-embedding call sites in `toy_example/train.py`. The regularizer can store detached generated targets or regenerate them from a previous hypernetwork and old embeddings, explicitly trading storage for computation. The inspected training path receives task indices; other source paths also contain a recognition model, so do not describe the entire repository as universally oracle-only. That recognition path was not reproduced or fully audited. Total memory must include embeddings, targets or predecessor weights and optimizer state, not only the shared hypernetwork. No numerical experiment from this repository was executed.

These sources constrain future latent memory designs. The remaining research task is to infer and assign latent state from actual observations, preserve useful distinctions under bounded compression and demonstrate the complete resource tradeoff. Combining established latent adaptation and generated weights is not a sufficient novelty distinction.

## Conditional regression is prior art

ALPaCA at `d06391a1bf11beda573078a6a1d62418aac367c4` already learns a neural
feature basis and prior for fast Bayesian linear regression. The inspected
`main/alpaca.py` computes posterior coefficients using context Gram/cross-product
matrices and trains on later query likelihood. Replacing raw cross-moments
with regularized coefficients, or adding learned features and a fast linear
head, cannot by itself establish novelty. This audit is source inspection,
not numerical reproduction. E22's learned multi-timescale gate also fails its
sufficient-repair claim despite actual online parameter adaptation.

Shared graph message passing, GRU state, input reinjection and parent decoding
are established neural algorithmic-reasoning methods. CLRS source was cloned and
inspected; Raven's shared recurrent core was revisited. The new 15,177-parameter
processor is a falsifiable component prototype, not an architectural novelty
claim. Neither source inspection nor an untrained functional preflight establishes
general reasoning or superior efficiency.
