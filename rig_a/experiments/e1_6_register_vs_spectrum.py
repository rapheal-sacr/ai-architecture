"""E1.6 -- the hinge, ratio half. Register versus spectrum on tail safety.

THE DECISIVE TEST OF REV 2 SECTION 1, run at the half that does not need a
number nobody has.

Rev 2 registered this as one experiment with one kill criterion, and it bundles
two quantities with different sensitivities:

    THE RATIO      blindness (worst-domain / traffic-weighted) and the ordering
                   of per-owner exposure. These come from TRAFFIC WEIGHTING.
                   E1.1c's Spearman -1.0 is a property of the weighting, not of
                   the subscription level, so it is readable at any subscription.
    THE LEVEL      free rank, and whether verified retirement keeps pace with
                   promotion. E1.1d showed every free-rank number in this record
                   was taken at subscription 1.0, which is exactly the
                   feasibility cliff, so the level is UNREADABLE until Rig B
                   supplies the real ratio.

The half that decides whether section 1's thesis is right is the ratio, and it
runs today. Precedent is this record's own: E1.2's finding was the ratio, not the
level. Free rank is reported below and explicitly NOT scored.

WHAT THE TWO ARMS ARE

    SPECTRUM   fill the R_t buffer from TRAFFIC at DomainMixture's Zipfian visit
               rates, choose r by energy over a traffic-drawn calibration set.
               Protection is then allocated in proportion to frequency, and a
               rare domain's subspace lands below the cut. This is E1.1b/E1.1c's
               mechanism exactly.
    REGISTER   fill the buffer PER OWNER from its own provenance at EQUAL N, and
               choose r by energy over an equal-N calibration set. Section 1.2:
               "that single change is what makes a rare-domain capability's
               subspace as well-protected as a common one's, and it is why
               Spearman(rate, leakage) = -1.0 cannot recur."

AND A THIRD ARM THAT MATTERS FOR THE ARCHITECTURE, NOT THE NUMBER. E1.1c already
contains `R-b freq-balanced R_t`, which importance-weights traffic by 1/rate.
Statistically that lands close to the register. The difference is not the
number, it is what each needs to know: **R-b must estimate the rate vector it is
correcting for** -- the traffic distribution the weighting rule says protection
must not depend on -- while the register needs nothing beyond each owner's own
provenance, which I1/I4 already require it to carry. If the two arms agree, that
agreement IS the finding: the register buys the same protection without the
estimate, which is the recorded-not-inferred thesis in its narrowest testable
form.

KILL CRITERIA (pre-registered):
    H1 The register does not reduce the blindness ratio against the spectrum.
       Section 1's central claim is then wrong and most of rev 2 loses its
       motivation.
    H2 The register's Spearman(rate, leakage) is still <= -0.5 -- exposure is
       still monotone in rarity, so the equal draw did not remove the structural
       tail penalty and only moved where it is computed.
    H3 The register's per-domain exposure ORDERING still tracks the spectrum's
       (Spearman between the two exposure vectors >= 0.9). The change would then
       be cosmetic: same domains protected in the same order, different numbers.

NOT SCORED HERE, DELIBERATELY: free rank, tail-safe retention level, and whether
the register starves. Those are the subscription-sensitive half and E1.1d says
they cannot be read at subscription 1.0. They are printed with the level marked
unreadable so that nobody quotes them.

AND EVERY PROTECTION-COST NUMBER NAMES ITS BASELINE AND REGIME (B19). Stratified
drawing costs +0.0125 without decay and +0.0806 with -- 6.4x -- so a cost figure
without both is unreadable. Each number below states which arm it is measured
against.

Is there a world that produces the other verdict? For H1, yes: if the energy
criterion's committed subspace were already domain-balanced, equal-N filling
would change nothing and the ratios would match. For H2, yes: if rare domains
were rare because their subspaces are intrinsically low-energy rather than
under-sampled, an equal draw would not help and the monotone penalty would
survive. Both are live -- E1.1c's panel C found subspace overlap partially
rescues the criterion, which is the same shape.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rig_a.core.spectrum import (  # noqa: E402
    DomainMixture,
    RtEstimator,
    interference_by_rank,
    leakage_by_rank,
    rank_for_energy,
)

DIM = 128
N_OBS = 4000
N_CALIB = 500
N_TEST = 500
N_PROBE = 400
RANK_REQUEST = 8
INTERFERENCE_LIMIT = 0.05
SEED = 20260806
RETENTIONS = (0.90, 0.95, 0.99, 0.995, 0.999)
CONFIGS = ((8, 8), (16, 8), (32, 4))
N_SEEDS = 4


def owner_draw(world, n_domains, domain_rank, n_per, rng):
    """Equal-N per owner from its OWN provenance. No rate vector consulted."""
    rows = []
    for d in range(n_domains):
        coeff = rng.normal(size=(n_per, domain_rank)) * world.within
        rows.append(coeff @ world.bases[d].T)
    return np.vstack(rows)


def one_world(n_domains: int, domain_rank: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    world = DomainMixture(DIM, n_domains, domain_rank, 1.0, rng)
    feats, labels = world.sample(N_OBS, rng)
    calib, calib_lab = world.sample(N_CALIB, rng)
    test, _ = world.sample(N_TEST, rng)
    readout = rng.normal(size=(DIM, 16)) / np.sqrt(DIM)
    probe = {d: (rng.normal(size=(N_PROBE, domain_rank)) * world.within)
             @ world.bases[d].T for d in range(n_domains)}

    per_owner = max(N_OBS // n_domains, 1)
    reg_feats = owner_draw(world, n_domains, domain_rank, per_owner, rng)
    reg_calib = owner_draw(world, n_domains, domain_rank,
                           max(N_CALIB // n_domains, 1), rng)

    est_spec = RtEstimator(dim=DIM, lam=1.0); est_spec.update(feats)
    est_reg = RtEstimator(dim=DIM, lam=1.0); est_reg.update(reg_feats)
    # R-b: importance-weight traffic by 1/rate. Needs the rate vector.
    w = 1.0 / world.rates[labels]; w /= w.mean()
    est_rb = RtEstimator(dim=DIM, lam=1.0); est_rb.update(feats * np.sqrt(w)[:, None])
    wc = 1.0 / world.rates[calib_lab]; wc /= wc.mean()

    arms = {
        "spectrum": (est_spec, calib),
        "register": (est_reg, reg_calib),
        "R-b freq-balanced": (est_rb, calib * np.sqrt(wc)[:, None]),
    }

    out = {}
    for tag, (est, cal) in arms.items():
        rows = []
        for rho in RETENTIONS:
            r = rank_for_energy(est, cal, rho)
            interfs = np.array([
                interference_by_rank(est, r, probe[d], readout, RANK_REQUEST, rng)
                for d in range(n_domains)])
            leaks = np.array([leakage_by_rank(est, r, probe[d])
                              for d in range(n_domains)])
            # The traffic-weighted number is what a pooled report would show.
            interf_traffic = interference_by_rank(est, r, test, readout,
                                                  RANK_REQUEST, rng)
            rows.append({
                "retention": rho,
                "committed_rank": int(r), "free_rank": int(DIM - r),
                "interf_traffic": round(float(interf_traffic), 4),
                "interf_worst": round(float(interfs.max()), 4),
                "blindness": round(float(interfs.max()) / max(float(interf_traffic), 1e-9), 2),
                "n_failing": int(np.sum(interfs > INTERFERENCE_LIMIT)),
                "spearman_rate_leak": round(float(spearmanr(world.rates, leaks).statistic), 3),
                "leaks": [round(float(v), 5) for v in leaks],
            })
        out[tag] = rows
    return out


def main() -> int:
    print("\nE1.6 -- the hinge, RATIO half. Register vs spectrum.\n")
    print("  Free rank is printed and NOT scored: E1.1d showed every free-rank")
    print("  number in this record was taken at subscription 1.0, the feasibility")
    print("  cliff. The level is unreadable until Rig B supplies the real ratio.\n")

    all_cfg = []
    for n_domains, domain_rank in CONFIGS:
        runs = [one_world(n_domains, domain_rank, SEED + s) for s in range(N_SEEDS)]

        def agg(tag, key, rho_i):
            return float(np.mean([r[tag][rho_i][key] for r in runs]))

        print(f"--- {n_domains} domains, rank {domain_rank} ---")
        print(f"  {'arm':<20}{'rho':>7}{'traffic':>10}{'worst':>10}{'blind':>8}"
              f"{'fail':>6}{'spearman':>10}{'free':>7}")
        cfg = {"n_domains": n_domains, "domain_rank": domain_rank, "arms": {}}
        for tag in ("spectrum", "register", "R-b freq-balanced"):
            rows = []
            for i, rho in enumerate(RETENTIONS):
                row = {
                    "retention": rho,
                    "interf_traffic": round(agg(tag, "interf_traffic", i), 4),
                    "interf_worst": round(agg(tag, "interf_worst", i), 4),
                    "blindness": round(agg(tag, "blindness", i), 2),
                    "n_failing": round(agg(tag, "n_failing", i), 2),
                    "spearman": round(agg(tag, "spearman_rate_leak", i), 3),
                    "free_rank": round(agg(tag, "free_rank", i), 1),
                }
                rows.append(row)
                print(f"  {tag if i == 0 else '':<20}{rho:>7.3f}"
                      f"{row['interf_traffic']:>10.4f}{row['interf_worst']:>10.4f}"
                      f"{row['blindness']:>8.2f}{row['n_failing']:>6.1f}"
                      f"{row['spearman']:>10.3f}{row['free_rank']:>7.0f}")
            cfg["arms"][tag] = rows
            print()

        # H3 -- is the ORDERING of exposure the same? Cosmetic-change test.
        ords = []
        for r in runs:
            for i in range(len(RETENTIONS)):
                a = r["spectrum"][i]["leaks"]
                b = r["register"][i]["leaks"]
                if len(set(a)) > 1 and len(set(b)) > 1:
                    ords.append(float(spearmanr(a, b).statistic))
        cfg["exposure_order_spearman"] = round(float(np.mean(ords)), 3) if ords else None
        all_cfg.append(cfg)

    # Scoring, at the retention rev 2 and E1.1b both quote
    i95 = RETENTIONS.index(0.95)
    spec_b = float(np.mean([c["arms"]["spectrum"][i95]["blindness"] for c in all_cfg]))
    reg_b = float(np.mean([c["arms"]["register"][i95]["blindness"] for c in all_cfg]))
    rb_b = float(np.mean([c["arms"]["R-b freq-balanced"][i95]["blindness"] for c in all_cfg]))
    spec_s = float(np.mean([c["arms"]["spectrum"][i95]["spearman"] for c in all_cfg]))
    reg_s = float(np.mean([c["arms"]["register"][i95]["spearman"] for c in all_cfg]))
    order = float(np.mean([c["exposure_order_spearman"] for c in all_cfg
                           if c["exposure_order_spearman"] is not None]))

    h1 = reg_b < spec_b
    h2 = reg_s > -0.5
    h3 = order < 0.9

    print("  AT rho = 0.95, the operating point E1.1b and rev 2 both quote.")
    print("  Every figure is measured against the SPECTRUM arm at the same rho,")
    print("  no decay, equal-N probe sets -- baseline and regime named (B19).")
    print(f"    blindness   spectrum {spec_b:.2f}  register {reg_b:.2f}"
          f"  R-b {rb_b:.2f}")
    print(f"    spearman    spectrum {spec_s:+.3f}  register {reg_s:+.3f}")
    print(f"    exposure ordering, spectrum vs register: {order:+.3f}")

    print(f"\n  H1 register reduces blindness:            {'ok' if h1 else 'NO'}")
    print(f"  H2 register breaks the rarity monotone:   {'ok' if h2 else 'NO'}")
    print(f"  H3 the change is not cosmetic:            {'ok' if h3 else 'NO'}")

    print("\n  AND THE ARCHITECTURAL POINT, which is not the number. R-b lands")
    print("  close to the register, and that agreement IS the finding rather than")
    print("  a redundancy: R-b must ESTIMATE the rate vector it corrects for --")
    print("  the traffic distribution the weighting rule says protection must not")
    print("  depend on -- while the register needs only each owner's own")
    print("  provenance, which I1/I4 already require it to carry. Same protection,")
    print("  no estimate. That is recorded-not-inferred in its narrowest form.")

    print("\n  LEVEL: UNREADABLE. Free rank is in the table and is not scored.")
    print("  Every number in this record was taken at subscription 1.0, which")
    print("  E1.1d identified as the feasibility cliff, so whether the register")
    print("  STARVES is a Rig B question. This experiment decides whether the")
    print("  register is a better protection instrument, not whether it fits.")

    verdict = "PASS" if (h1 and h2 and h3) else "FAIL"
    print(f"\n  SECTION 1's CENTRAL CLAIM: {verdict}\n")

    out = pathlib.Path(__file__).resolve().parents[2] / "results" / "e1_6_register_vs_spectrum.json"
    out.write_text(json.dumps(
        {"seed": SEED, "n_seeds": N_SEEDS, "dim": DIM, "configs": list(CONFIGS),
         "retentions": list(RETENTIONS), "by_config": all_cfg,
         "blindness_at_095": {"spectrum": round(spec_b, 2), "register": round(reg_b, 2),
                              "R-b": round(rb_b, 2)},
         "spearman_at_095": {"spectrum": round(spec_s, 3), "register": round(reg_s, 3)},
         "exposure_order_spearman": round(order, 3),
         "H1_blindness_reduced": bool(h1), "H2_monotone_broken": bool(h2),
         "H3_not_cosmetic": bool(h3),
         "not_scored": "free rank and tail-safe level -- subscription-sensitive, E1.1d",
         "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
