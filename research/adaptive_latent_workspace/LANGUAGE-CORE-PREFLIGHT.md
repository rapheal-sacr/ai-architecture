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

The local configuration has 24 layers, hidden width 896, 14 query heads and two
KV heads; maximum position embeddings is configured as 32,768. That last value
is not a measured long-context capability. CPU float32/eager inference reports
494,032,768 unique parameters, 1,976,131,072 parameter bytes and tied input/output
embedding storage. Each 54-token prompt produces 1,327,104 KV-cache bytes.
Peak process RSS in the preflight is about 3.42 GB and includes loading overhead.

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

## GPU adaptation feasibility

After E29 terminated, a separate preflight loaded the base in float16 on the
GTX1060 6GB and inserted rank 4 low-rank updates into q/v projections in all 24
layers. This is conventional low-rank adaptation, not an invention. There are
270,336 trainable float32 parameters (1,081,344 bytes), in addition to 988,065,536
frozen base parameter bytes. Zero adapters preserve original logits exactly.
Two 45-token repeated-example Adam updates have finite gradients and reduce
same-example loss from 0.659 to 0.069. This is a functionality check, not transfer.

The base-weight hash is unchanged; saved adapters restore trained logits exactly.
Two updates take 0.389 seconds in this preflight and peak allocated CUDA tensor
memory is 1,234,241,024 bytes. These numbers exclude untracked driver/context and
Python overhead and say nothing about large-context training. Optimizer-state
restoration across a real continual run, old-task retention, independent memory
queries, compression and procedure self-improvement remain untested here.
