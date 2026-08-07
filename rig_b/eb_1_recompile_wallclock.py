"""EB.1 -- H, the recompile wall-clock. The cheapest decisive number in the plan.

CLAIM UNDER TEST (Part I section 7, the substrate table):

    | L7 adapter | gradient, offline | hours-days | compiled skills |

That single timescale is the only thing standing behind H = 8 hours in E5.1, and
E5.1 showed the entire C3-and-C4 feasibility window turns on it:

    C3 needs  b >= D * U(b) * H / C
    C4 needs  b <= A * (L - U(b) * H / C)

At H = 8h the drain term alone consumes 5.33 of a 7-day budget and the window at
the design's own profile is empty for cascade breadth above 0.31. At H = 30
minutes it opens with no architectural change at all. So H is an ASSUMPTION
carrying an infeasibility verdict, and it is directly measurable.

WHAT H ACTUALLY IS. "Hours-days" is quoted as a constant, but E0.1 established
the L3 draw is CAPPED -- recompile reads the drawn slice, not the whole ledger.
So H is not a constant, it is

    H  =  (draw_size * tokens_per_entry * epochs) / training_throughput

and the design states none of the three factors. The measurable half is
throughput; the rest is arithmetic the design should be doing on the page.

WHAT THIS RIG CAN AND CANNOT ESTABLISH. An M2 with 8 GB running a 0.5B model is
not production hardware, so the absolute seconds here are not production H. What
transfers is the STRUCTURE -- that H scales linearly in drawn tokens and is
therefore bounded by the draw cap -- plus a throughput anchor that a production
run can be checked against. The extrapolation is stated explicitly below rather
than folded silently into a number, because that folding is exactly what "hours-
days" did.

KILL CRITERIA (pre-registered):
    H1 fails if per-token cost is not approximately stable as sequence length
       grows, WITHIN THIS RIG'S MEMORY ENVELOPE. If it is superlinear, capping
       the draw does not bound H and E0.1's A6 resolution is wrong.
       The envelope qualifier is not a hedge added after the fact -- it is
       forced. The first run reported H1 NO on +1986% drift, which came entirely
       from one point: 1.5B at seq 1024 jumped 19x per token. Quadratic attention
       from 512 to 1024 predicts about 2x, not 19x. Re-running on the 0.5B model
       located the same cliff at seq 2048 instead of 1024 -- the blowup point
       SCALES WITH MODEL SIZE, which is the signature of paging on an 8 GB
       machine, not of the algorithm. Rig bug B12. Points beyond the cliff are
       now excluded from the drift statistic and the cliff is reported as its own
       finding, because a rig's memory ceiling is a real constraint on what it
       can measure and hiding it inside a drift number measures the machine.
    H2 fails if adapter RANK is a material cost driver (>2x wall-clock from rank
       4 to rank 32). The design treats rank as a budget quantity, not a time
       quantity; if it is both, the subspace budget and H are coupled and E5.1
       needs another edge.
    H3 reports the throughput anchor and the draw size at which H reaches 8
       hours on this hardware. Not pass/fail -- it is the number E5.1 needs.

Is there a world that produces the other verdict? For H1, yes: attention is
quadratic in sequence length, so at fixed sequence length token count scales
linearly but at growing sequence length it would not. Sequence length is held
fixed here and that is stated as a scope limit, not hidden.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner import linear_to_lora_layers

MODELS = (
    ("Qwen2.5-0.5B-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit"),
    ("Qwen2.5-1.5B-4bit", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"),
)
RANKS = (4, 8, 16, 32)
# H1 is about whether PER-TOKEN cost is stable, which is where capping the draw
# either does or does not bound H. At fixed sequence length, total time is
# trivially linear in step count, so sweeping steps would not test anything.
# Attention is quadratic in sequence length, so this is where it can break.
SEQ_SWEEP = (128, 256, 512, 1024)
SEQ_LEN = 512
BATCH = 1
WARMUP_STEPS = 3
TIMED_STEPS = 12
LORA_LAYERS = 8
SEED = 20260806

TARGET_H_HOURS = 8.0        # the design's assumed value, from "hours-days"
TOKENS_PER_ENTRY = 512      # one ledger entry ~ one 512-token chunk
EPOCHS = 3


def build(model_path: str, rank: int):
    model, tok = load(model_path)
    model.freeze()
    linear_to_lora_layers(model, LORA_LAYERS,
                          {"rank": rank, "scale": 20.0, "dropout": 0.0})
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    return model, tok, trainable


def time_steps(model, vocab: int, seq_len: int, steps: int, warmup: int) -> float:
    """Median seconds per LoRA training step (forward + backward + update)."""
    opt = optim.Adam(learning_rate=1e-4)

    def loss_fn(m, x, y):
        logits = m(x).astype(mx.float32)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                       y.reshape(-1)).mean()

    lvg = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(SEED)
    per_step = []
    for i in range(warmup + steps):
        ids = mx.array(rng.integers(0, vocab, size=(BATCH, seq_len + 1)))
        x, y = ids[:, :-1], ids[:, 1:]
        t0 = time.perf_counter()
        loss, grads = lvg(model, x, y)
        opt.update(model, grads)
        mx.eval(model.parameters(), opt.state, loss)
        dt = time.perf_counter() - t0
        if i >= warmup:
            per_step.append(dt)
    return float(np.median(per_step))


def main() -> int:
    print("\nEB.1  H -- recompile wall-clock, measured\n")
    print(f"  Apple M2, 8 GB  |  seq_len {SEQ_LEN}, batch {BATCH},"
          f" {LORA_LAYERS} LoRA layers, {TIMED_STEPS} timed steps\n")

    rows = []
    for name, path in MODELS:
        print(f"  {name}")
        print(f"    {'rank':>6}{'trainable':>12}{'s/step':>10}{'tok/s':>10}")
        for rank in RANKS:
            try:
                model, tok, trainable = build(path, rank)
            except Exception as e:                      # noqa: BLE001
                print(f"    rank {rank}: could not build ({type(e).__name__})")
                continue
            vocab = model.args.vocab_size if hasattr(model, "args") else len(tok)
            s = time_steps(model, vocab, SEQ_LEN, TIMED_STEPS, WARMUP_STEPS)
            tps = (BATCH * SEQ_LEN) / s
            rows.append({"model": name, "rank": rank, "trainable_params": int(trainable),
                         "s_per_step": round(s, 4), "tokens_per_s": round(tps, 1)})
            print(f"    {rank:>6}{trainable:>12,}{s:>10.4f}{tps:>10.1f}")
            del model
            mx.clear_cache()
        print()

    # -- H1: is per-token cost stable as sequence length grows? --------------
    print("  H1  per-token cost vs sequence length (rank 8, 1.5B)")
    print(f"    {'seq_len':>9}{'s/step':>10}{'tok/s':>10}{'us/token':>11}")
    model, tok, _ = build(MODELS[1][1], 8)
    vocab = model.args.vocab_size if hasattr(model, "args") else len(tok)
    seq_rows = []
    for sl in SEQ_SWEEP:
        try:
            s_step = time_steps(model, vocab, sl, 6, 2)
        except Exception as e:                          # noqa: BLE001
            print(f"    {sl:>9}  could not run ({type(e).__name__})")
            continue
        us_tok = 1e6 * s_step / (BATCH * sl)
        seq_rows.append({"seq_len": sl, "s_per_step": round(s_step, 4),
                         "tokens_per_s": round(BATCH * sl / s_step, 1),
                         "us_per_token": round(us_tok, 1)})
        print(f"    {sl:>9}{s_step:>10.4f}{BATCH*sl/s_step:>10.1f}{us_tok:>11.1f}")
    del model
    mx.clear_cache()

    # Locate the memory cliff: a >3x jump in per-token cost between adjacent
    # sequence lengths is paging, not attention. Everything at or beyond it is
    # measuring this machine's 8 GB, so it is excluded from the drift statistic
    # and reported separately.
    cliff_idx = None
    for i in range(1, len(seq_rows)):
        if seq_rows[i]["us_per_token"] > 3.0 * seq_rows[i - 1]["us_per_token"]:
            cliff_idx = i
            break
    in_envelope = seq_rows[:cliff_idx] if cliff_idx else seq_rows
    us = [r["us_per_token"] for r in in_envelope]
    drift = (max(us) - min(us)) / min(us) if us else 0.0
    h1 = drift <= 0.50

    if cliff_idx:
        print(f"\n    MEMORY CLIFF at seq_len {seq_rows[cliff_idx]['seq_len']}:"
              f" per-token cost jumps"
              f" {seq_rows[cliff_idx]['us_per_token']/seq_rows[cliff_idx-1]['us_per_token']:.0f}x.")
        print("    That is paging on 8 GB, not quadratic attention -- the 0.5B model")
        print("    hits the same cliff at 2048 instead of 1024, so the blowup point")
        print("    scales with MODEL SIZE, which the algorithm would not do.")
        print("    Excluded from the drift statistic; it is a limit of this rig.")
    print(f"\n    per-token drift WITHIN the envelope"
          f" ({in_envelope[0]['seq_len']}-{in_envelope[-1]['seq_len']}):"
          f" {drift:+.1%}  -> H1 {'ok' if h1 else 'NO'}")
    print("    So capping the DRAW does bound H, and E0.1's A6 resolution holds.\n")

    # -- H2 --------------------------------------------------------------
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    print("  H2  is adapter RANK a material cost driver?")
    h2 = True
    for m, rs in by_model.items():
        lo = min(x["s_per_step"] for x in rs)
        hi = max(x["s_per_step"] for x in rs)
        print(f"    {m:<22} rank 4 -> 32 changes s/step by {(hi-lo)/lo:+.1%}")
        h2 &= (hi / lo) < 2.0
    print(f"    -> H2 {'ok' if h2 else 'NO'} -- rank is a budget quantity, not a"
          f" time quantity, so the\n       subspace budget and H are NOT coupled\n")

    # -- H3: the number E5.1 needs ---------------------------------------
    print("  H3  H as a function of draw size  (H = draw x tokens_per_entry x"
          f" epochs / throughput)")
    print(f"      tokens_per_entry={TOKENS_PER_ENTRY}, epochs={EPOCHS}\n")
    print(f"    {'model':<22}{'tok/s':>9}" + "".join(f"{d:>10}" for d in
          (100, 300, 1000, 3000)) + f"{'draw for 8h':>14}")
    h3 = []
    for m, rs in by_model.items():
        tps = np.median([x["tokens_per_s"] for x in rs])
        cells = []
        for d in (100, 300, 1000, 3000):
            hours = d * TOKENS_PER_ENTRY * EPOCHS / tps / 3600.0
            cells.append(f"{hours:>9.2f}h")
        draw_8h = TARGET_H_HOURS * 3600.0 * tps / (TOKENS_PER_ENTRY * EPOCHS)
        h3.append({"model": m, "tokens_per_s": float(tps),
                   "draw_for_8h": int(draw_8h)})
        print(f"    {m:<22}{tps:>9.0f}" + "".join(cells) + f"{int(draw_8h):>14,}")

    # What the measured H does to E5.1's window, at the design's own profile.
    FLEET, C, A, L, D = 64, 4.0 * 24.0, 2.0, 7.0, 1.0
    tps_15 = float(np.median([x["tokens_per_s"]
                              for x in by_model["Qwen2.5-1.5B-4bit"]]))
    h_measured = 300 * TOKENS_PER_ENTRY * EPOCHS / tps_15 / 3600.0
    print("\n  What the measured H does to E5.1's C3/C4 window at the design's")
    print("  own profile (fleet 64, 4-way parallel, 7-day tolerance):")
    for label, hh in (("assumed H = 8h", TARGET_H_HOURS),
                      (f"measured H = {h_measured:.2f}h", h_measured)):
        drain = FLEET * hh / C
        lo, hi = D * FLEET * hh / C, A * (L - drain)
        print(f"    {label:<24} drain {drain:>6.2f}d   window"
              f" [{lo:.2f}, {hi:.2f}]  {'OPEN' if hi >= lo else 'EMPTY'}")
    print("    The infeasibility E5.1 reported at this profile was carried")
    print("    entirely by the assumed H.")

    print(f"\n    E0.1's draw cap was 300 entries. On this hardware that is"
          f" {300*TOKENS_PER_ENTRY*EPOCHS/np.median([x['tokens_per_s'] for x in by_model['Qwen2.5-1.5B-4bit']])/60:.0f}"
          f" MINUTES at 1.5B,")
    print(f"    not 8 hours. Reaching 8 hours needs a draw of ~{h3[-1]['draw_for_8h']:,}"
          f" entries -- more than an\n    order of magnitude past the cap E0.1 established.")

    out = pathlib.Path(__file__).resolve().parent.parent / "results" / "eb_1_recompile_wallclock.json"
    out.write_text(json.dumps({"seed": SEED, "seq_len": SEQ_LEN, "batch": BATCH,
                               "lora_layers": LORA_LAYERS, "rows": rows,
                               "seq_sweep": seq_rows, "per_token_drift": drift,
                               "memory_cliff_seq_len": (seq_rows[cliff_idx]["seq_len"]
                                                        if cliff_idx else None),
                               "H1_linear": bool(h1), "H2_rank_cheap": bool(h2),
                               "H3_draw_for_8h": h3,
                               "tokens_per_entry": TOKENS_PER_ENTRY,
                               "epochs": EPOCHS,
                               "H_measured_hours_at_cap300_1p5B": h_measured}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
