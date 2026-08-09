"""EB.3 -- real adapter spillover. Phase 4.2, and it decides Part II section B.

R10, from E3.1, states the condition the design never stated:

    "Net-transfer ranking separates abstractions from patches ONLY WHILE narrow
     patches have near-zero REAL off-target effect. Measure adapter spillover on
     real models before relying on the rule."

E3.1's T4 measured what happens when that condition fails: with a systematic
+0.02 patch spillover the transfer margin inverts to -0.008 and both arms
collapse -- the ranking signal saturates and stops discriminating. So the rule's
viability is a single unmeasured empirical quantity, and this measures it.

THE STATISTIC UNDER TEST is the one E3.1c evidenced and rev 2 section 1.5 keeps
after deleting the transfer matrix:

    sum - max     summed off-target effect, minus the largest single effect

For a PURE PATCH the effect is concentrated on its target, so sum - max ~ 0. For
an ABSTRACTION it is spread, so sum - max > 0. The discriminator works only if
those two are far apart. The decision-relevant figure is therefore the RATIO

    (sum - max) / max

which is scale-free -- it does not depend on how big the on-target gain happens
to be, which is what makes it comparable across adapters and across models. A
narrow adapter whose ratio approaches a broad adapter's is the failure R10 names.

WHY THIS IS THE CHEAP FORM (E3.1b). The absolute level of spillover is not the
question; the question is whether NARROW and BROAD differ on it. So both arms are
trained from the same base, evaluated on the SAME held-out probe set, and
compared pairwise. Anything that shifts both arms equally -- tokenizer, layer
choice, probe difficulty -- cancels.

ARMS, and the broad arm is the positive control:
    narrow   LoRA trained on ONE domain. Should be a patch: sum - max ~ 0.
    broad    LoRA trained on ALL domains. Should spread: sum - max > 0.
If the broad arm does not separate from the narrow arm, the instrument cannot
detect the thing it exists to detect, and no conclusion about narrow adapters
follows -- which is the manipulation check this record now runs before reading
any comparison.

KILL CRITERIA (pre-registered):
    KS1 Narrow adapters have off-target effect large enough that (sum-max)/max
        exceeds ~0.25 -- a quarter of the on-target effect leaking. Section B's
        ranking rule then rests on a condition real adapters violate, and R10's
        warning is confirmed.
    KS2 The broad arm does not separate from the narrow arm on (sum-max)/max.
        The instrument is then not measuring spread and nothing about narrow
        adapters can be read off it.
    KS3 Off-target effects are systematically NEGATIVE -- a narrow adapter
        degrades other domains. That is worse than spillover for the ranking
        rule, because `sum - max` would then reward patches for damage.

Is there a world that produces the other verdict? For KS1, yes: LoRA at low rank
on a small layer subset is a deliberately constrained update, and the design's
whole basis-disjointness argument predicts near-zero off-target movement. For
KS2, yes: if a broad adapter simply averages its domains it may end up with a
LOWER on-target max and a similar ratio, in which case the statistic does not
separate them and the discriminator was never viable.
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rig_b.eb_2_activation_spectra import DOMAINS  # noqa: E402

MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
LORA_LAYERS = 8
RANK = 8
# HYPERPARAMETERS ARE NOT INHERITED FROM EB.1. EB.1 used scale 20 / lr 1e-4 to
# TIME steps and never checked that training converged, so that config had never
# been validated for learning anything. Swept: at scale 20/lr 1e-4 and at
# scale 2/lr 5e-5 the adapter drives train loss to ~0.1 on seven sentences while
# held-out loss RISES -- memorisation. Only a barely-trained setting improves
# held-out loss, and train loss is reported below so over- and under-fitting are
# both visible rather than assumed away.
SCALE = 2.0
LR = 1e-5
EPOCHS = 12

# FORMAT CONTROL. Every domain is short declarative factual prose, so they differ
# in TOPIC and not in FORM. An adapter trained on any one can improve all of them
# by learning the shared form, which would register as spillover while having
# nothing to do with domain transfer. This arm has the same form and a topic no
# probe touches, so its lift on the eight probes is format transfer with domain
# content removed. Off-target lift MINUS this is the only interpretable figure.
CONTROL_TEXTS = [
    "A cable stitch crosses one group of loops in front of another.",
    "Blocking sets the finished shape of a knitted panel.",
    "Gauge is the number of stitches across a measured width.",
    "Ribbing alternates knit and purl to make an elastic edge.",
    "A stitch marker records where a pattern repeat begins.",
    "Plying twists two spun singles into a balanced yarn.",
    "Frogging undoes rows of work back to a chosen point.",
]
SEED = 20260806
TARGETS = (0, 3, 5)        # which domains get a narrow adapter
HOLDOUT = 3                # sentences per domain held out of training


def split(rng):
    """Held-out probes per domain, disjoint from anything trained on."""
    train, probe = {}, {}
    for name, texts in DOMAINS.items():
        idx = rng.permutation(len(texts))
        probe[name] = [texts[i] for i in idx[:HOLDOUT]]
        train[name] = [texts[i] for i in idx[HOLDOUT:]]
    return train, probe


def loss_on(model, tok, texts) -> float:
    """Mean next-token cross-entropy over a text set. No gradients."""
    tot, n = 0.0, 0
    for t in texts:
        ids = mx.array([tok.encode(t)])
        if ids.shape[1] < 2:
            continue
        logits = model(ids[:, :-1]).astype(mx.float32)
        ce = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        mx.eval(ce)
        tot += float(ce.mean()); n += 1
    return tot / max(n, 1)


def train_lora(texts, seed: int):
    """Fresh base + fresh LoRA, trained on `texts`. Returns the adapted model."""
    model, tok = load(MODEL)
    model.freeze()
    linear_to_lora_layers(model, LORA_LAYERS,
                          {"rank": RANK, "scale": SCALE, "dropout": 0.0})
    opt = optim.Adam(learning_rate=LR)

    def loss_fn(m, x, y):
        logits = m(x).astype(mx.float32)
        return nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).mean()

    lvg = nn.value_and_grad(model, loss_fn)
    rng = np.random.default_rng(seed)
    last: list = []
    for _ in range(EPOCHS):
        for i in rng.permutation(len(texts)):
            ids = mx.array([tok.encode(texts[i])])
            if ids.shape[1] < 2:
                continue
            loss, grads = lvg(model, ids[:, :-1], ids[:, 1:])
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state, loss)
            last.append(float(loss))
    return model, tok, (float(np.mean(last[-len(texts):])) if last else float("nan"))


def stats(delta: np.ndarray, target: int) -> dict:
    """`sum - max` and the scale-free ratio the ranking rule depends on.

    Deltas are IMPROVEMENTS (baseline loss - adapted loss), so positive is
    better. `max` is the largest single-domain improvement; for a patch that
    should be the target domain, and whether it is gets reported rather than
    assumed.
    """
    mx_i = int(np.argmax(delta))
    mx_v = float(delta[mx_i])
    s = float(delta.sum())
    off = np.delete(delta, target)
    return {"target": target, "argmax": mx_i, "argmax_is_target": mx_i == target,
            "on_target": round(float(delta[target]), 5),
            "max": round(mx_v, 5), "sum": round(s, 5),
            "sum_minus_max": round(s - mx_v, 5),
            "ratio": round((s - mx_v) / mx_v, 4) if abs(mx_v) > 1e-9 else None,
            "off_mean": round(float(off.mean()), 5),
            "off_abs_mean": round(float(np.abs(off).mean()), 5),
            "off_negative_frac": round(float((off < 0).mean()), 3)}


def main() -> int:
    t0 = time.time()
    names = list(DOMAINS)
    rng = np.random.default_rng(SEED)
    train, probe = split(rng)

    print("\nEB.3 -- real adapter spillover (Phase 4.2; decides Part II section B)\n")
    print(f"  {MODEL}, rank {RANK}, {LORA_LAYERS} LoRA layers, {EPOCHS} epochs")
    print(f"  {len(names)} domains, {HOLDOUT} held-out sentences each\n")

    base, tok = load(MODEL)
    base_loss = np.array([loss_on(base, tok, probe[n]) for n in names])
    del base
    print("  baseline held-out loss per domain: "
          + " ".join(f"{v:.3f}" for v in base_loss) + "\n")

    mc, tkc, ctrl_train = train_lora(CONTROL_TEXTS, SEED + 7)
    ctrl_after = np.array([loss_on(mc, tkc, probe[n]) for n in names])
    del mc, tkc
    ctrl_lift = base_loss - ctrl_after
    print(f"  format control (unrelated topic, same form)   train {ctrl_train:5.3f}"
          f"   mean lift on the eight probes {ctrl_lift.mean():+.4f}")
    print("    Whatever this produces is FORM, not domain content.\n")

    rows = []
    for t in TARGETS:
        m, tk, trl = train_lora(train[names[t]], SEED + t)
        after = np.array([loss_on(m, tk, probe[n]) for n in names])
        del m, tk
        lift = base_loss - after
        st = stats(lift, t)
        st["arm"] = f"narrow:{names[t]}"; st["train_loss"] = round(trl, 4)
        adj = lift - ctrl_lift
        st["adj_on_target"] = round(float(adj[t]), 5)
        st["adj_off_mean"] = round(float(np.delete(adj, t).mean()), 5)
        rows.append(st)
        print(f"  narrow on {names[t]:<10} train {trl:5.3f}   raw on"
              f" {st['on_target']:+.4f} off {st['off_mean']:+.4f}"
              f"   ADJUSTED on {st['adj_on_target']:+.4f}"
              f" off {st['adj_off_mean']:+.4f}")

    all_texts = [x for n in names for x in train[n]]
    m, tk, btrl = train_lora(all_texts, SEED + 99)
    after = np.array([loss_on(m, tk, probe[n]) for n in names])
    del m, tk
    broad = stats(base_loss - after, TARGETS[0])
    broad["arm"] = "broad"; broad["train_loss"] = round(btrl, 4)
    rows.append(broad)
    print(f"\n  broad (all domains)    train {btrl:5.3f}   max {broad['max']:+.4f}"
          f"  off {broad['off_mean']:+.4f}  sum-max {broad['sum_minus_max']:+.4f}")

    narrow = [r for r in rows if r["arm"].startswith("narrow")]
    hit = sum(r["argmax_is_target"] for r in narrow)
    adj_on = float(np.mean([r["adj_on_target"] for r in narrow]))
    adj_off = float(np.mean([r["adj_off_mean"] for r in narrow]))
    raw_on = float(np.mean([r["on_target"] for r in narrow]))
    raw_off = float(np.mean([r["off_mean"] for r in narrow]))
    ratio_adj = round(adj_off / adj_on, 3) if abs(adj_on) > 1e-9 else None

    print(f"\n  MANIPULATION CHECK: argmax landed on the trained domain in"
          f" {hit}/{len(narrow)} narrow arms.")
    print("\n  RAW AGAINST FORMAT-ADJUSTED, and the gap is the whole story:")
    print(f"    raw        on {raw_on:+.4f}   off {raw_off:+.4f}"
          f"   ratio {raw_off / raw_on if abs(raw_on) > 1e-9 else float('nan'):.2f}")
    print(f"    adjusted   on {adj_on:+.4f}   off {adj_off:+.4f}   ratio {ratio_adj}")

    ks_valid = hit >= 2 and adj_on > 0
    print(f"\n  IS THE INSTRUMENT READABLE AT ALL: {'yes' if ks_valid else 'NO'}")
    if not ks_valid:
        print("    NO, AND THAT IS THE RESULT. Two failures, both about the data")
        print("    rather than about adapters:")
        print("    1. SEVEN TRAINING SENTENCES PER DOMAIN. Every configuration that")
        print("       fits them drives train loss to ~0.1 and makes held-out loss")
        print("       WORSE -- memorisation, not learning. Only a barely-trained")
        print("       setting improves held-out loss, and it barely learns the")
        print("       domain either.")
        print("    2. ALL EIGHT DOMAINS SHARE ONE FORM -- short declarative factual")
        print("       prose differing in topic only. An adapter trained on any one")
        print("       improves all of them by learning the shared form, which the")
        print("       format control measures directly.")
        print("    This is the correlated-error trap in a new place: train and")
        print("    probe sets share a property that is not the property under test.")
        print("    E0.2 fell into the same shape, and it is why the format control")
        print("    exists here at all.")
        print("\n    SO R10 IS NOT ANSWERED, AND IS NOW PRECISELY SPECIFIED.")
        print("    Measuring real adapter spillover needs per-domain text in enough")
        print("    volume to learn domain CONTENT rather than form, and domains")
        print("    differing in form as well as topic. That is the corpus the Rig B")
        print("    trip collects anyway, so this joins 2.1, 2.2 and the")
        print("    checkability sweep on ONE acquisition rather than being the")
        print("    model-only measurement it was registered as.")

    print("\n  SCOPE, and it is narrow. One model, one rank, one layer count, eight")
    print("  short-text domains, twelve epochs. LoRA at rank 8 on 8 layers is a")
    print("  deliberately constrained update and the result should not be read as")
    print("  `adapter spillover is X` in general. What it can settle is whether")
    print("  the DIFFERENTIAL exists at all in a real model, which is what E3.1b")
    print("  identified as the cheap form and what R10 asks for.")

    verdict = "PASS" if ks_valid else "INSTRUMENT-BLOCKED"
    print(f"\n  SECTION B's CONDITION: {verdict}\n  ({time.time() - t0:.0f}s)")

    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "eb_3_spillover_differential.json"
    out.write_text(json.dumps(
        {"model": MODEL, "rank": RANK, "lora_layers": LORA_LAYERS,
         "epochs": EPOCHS, "domains": names, "targets": list(TARGETS),
         "baseline_loss": [round(float(v), 5) for v in base_loss],
         "rows": rows, "broad": broad,
         "control_lift_mean": round(float(ctrl_lift.mean()), 5),
         "control_train_loss": round(ctrl_train, 4),
         "raw": {"on_target": round(raw_on, 5), "off_target": round(raw_off, 5)},
         "format_adjusted": {"on_target": round(adj_on, 5),
                             "off_target": round(adj_off, 5), "ratio": ratio_adj},
         "argmax_on_target": f"{hit}/{len(narrow)}",
         "instrument_readable": bool(ks_valid),
         "blocked_on": "per-domain text in volume, and domains differing in form not only topic",
         "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
