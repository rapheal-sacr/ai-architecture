# Language-core feasibility — no continual-learning result

The graph processor remains a component assay. A minimal integrated test will
also need language interpretation, memory access and persistent adaptation.
This feasibility preparation does not validate that architecture or replace the
full goal with two easy language queries.

Cloned the Qwen2.5-0.5B-Instruct model repository at
`7ae557604adf67be50417f59c2c2f167def9a775`. The 988,097,824-byte safetensors
artifact was retrieved at that exact commit and checked against the SHA256 in
the cloned Git LFS pointer:
`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`.
The original clone remains untouched; the executable snapshot is under WD
`models/qwen2.5-0.5b-instruct-7ae5576`. The repository declares Apache-2.0 and the
license is preserved. No rendered model page was fetched.

Cloned Transformers v4.46.3 at
`052e652d6d53c2b26ffde87e039b723949a53493`; inspected the Qwen2 attention
projection, rotary-position, grouped-KV/cache and causal-LM paths alongside the
model configuration. A separate WD `language-runtime` installs that source and
shares the existing torch/numpy packages read-only through a `.pth` entry. Its
additional dependencies do not change the ongoing graph experiment's runtime.

The local configuration has 24 layers, hidden width896, 14 query heads and two
KV heads; maximum position embeddings is configured as32,768. That last value
is not a measured long-context capability. CPU float32/eager inference reports
494,032,768 unique parameters, 1,976,131,072 parameter bytes and tied input/output
embedding storage. Each 54-token prompt produces 1,327,104 KV-cache bytes.
Peak process RSS in the preflight is about3.42GB and includes loading overhead.

Two prompts have identical token multisets but reverse red/blue box order.
Their logits differ, and greedy outputs are respectively blue and red. This
checks basic order sensitivity and execution, not broad reasoning, persistent
memory, new learning, cost superiority or architectural novelty. All execution
uses local files with remote custom code disabled and network access disabled
in the model loader. The GPU was left to E29; GPU feasibility and adaptation
remain unmeasured.

Before any integrated experiment: verify GPU/adapter execution and full costs,
define fresh closed-loop tasks and strong explicit-memory controls, freeze the
protocol, and separate pretrained competence from acquired facts and changed
procedures. Existing pretrained ability cannot count as this architecture's
self-improvement. Retain all prior failure obligations in `GOAL-EVIDENCE.md`.
