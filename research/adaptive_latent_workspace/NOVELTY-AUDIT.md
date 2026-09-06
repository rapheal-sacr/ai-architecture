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
