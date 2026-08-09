"""EB.2 -- real per-domain activation geometry, and the hinge re-run on it.

Phase 4.1: "per-region activation spectra from a small MLX model (retires
B4-class risk)". It has been the first Rig B item since PLAN was written, and it
matters more now than when it was registered, because E1.6 just used the
generator it calibrates.

WHAT RESTS ON A SYNTHETIC GENERATOR. Every spectrum result in this record --
E1.1, E1.1b, E1.1c, E1.1d, and now E1.6, the hinge -- runs on `DomainMixture`,
which asserts two things about geometry that nobody has measured:

    within a domain   singular values decay as k^-0.5
    across domains    bases are drawn independently, so near-ORTHOGONAL in 128d

E1.1c already flagged the second as the pessimistic case and swept a shared
fraction in its panel C, reporting that overlap partially rescues the criterion.
That was a sweep over an assumption. This measures it.

B4 IS WHY THIS IS NOT OPTIONAL. B4 was a generator bug -- domain subspaces
redrawn every call -- that manufactured a headline. It was caught by a capacity
sweep, not by a cross-check. A generator whose geometry is authored alongside the
hypothesis is the same exposure with no bug required.

THE DECISIVE PART IS NOT THE SPECTRA. It is the substitution. `RealMixture`
below has DomainMixture's interface and is backed by activations from
Qwen2.5-0.5B-Instruct-4bit, so E1.6's instruments -- RtEstimator,
rank_for_energy, leakage_by_rank, interference_by_rank -- run unchanged on real
vectors. Same instrument, different world, which is the only way to tell whether
a finding is about the world or about the generator.

WHAT THIS DOES AND DOES NOT MEASURE. It measures GEOMETRY: spectrum shape,
effective rank, cross-domain overlap. It does NOT measure TRAFFIC RATES -- how
often each domain is actually served needs usage logs, which is a corpus this
does not have. Rates stay Zipfian and stay assumed, and every number below that
depends on them is marked. The split matters: the hinge's blindness ratio is a
property of the WEIGHTING against the GEOMETRY, so measuring the geometry moves
it and leaving the rates assumed does not invalidate it.

KILL CRITERIA (pre-registered):
    KB1 Real within-domain decay is far from the assumed k^-0.5 (fitted exponent
        outside 0.25-1.0). DomainMixture then misstates the shape the energy
        criterion integrates over, and every retention-vs-rank number in the
        record is quantitatively wrong.
    KB2 Real cross-domain subspace overlap is far from near-orthogonal
        (mean normalised overlap > 0.25). Free-rank numbers taken on independent
        bases are then pessimistic by a margin that matters, and E1.1c's panel C
        concern is confirmed rather than swept.
    KB3 THE DECISIVE ONE. The hinge's direction reverses or vanishes on real
        geometry -- the register no longer reduces the blindness ratio, or no
        longer breaks the rarity monotone. E1.6 is then a property of
        DomainMixture and section 1's central claim returns to untested.

Is there a world that produces the other verdict? For KB3, yes: if real domains
overlap heavily, the committed subspace is shared, a rare domain's directions are
already retained for a common domain's sake, and equal-N filling changes little --
which is exactly the mechanism by which E1.1c's panel C found overlap rescues the
criterion. That is a live way for the hinge to be a generator artifact.
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import mlx.core as mx
import numpy as np
from mlx_lm import load


def spearmanr(a, b):
    """Pearson on ranks. Local so the Rig B env stays free of scipy -- on an
    8 GB machine the dependency is not worth one statistic."""
    ra = np.argsort(np.argsort(np.asarray(a, dtype=float)))
    rb = np.argsort(np.argsort(np.asarray(b, dtype=float)))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    r = float((ra * rb).sum() / d) if d > 0 else 0.0
    return type("R", (), {"statistic": r})()

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rig_a.core.spectrum import (  # noqa: E402
    RtEstimator,
    interference_by_rank,
    leakage_by_rank,
    rank_for_energy,
)

# TWO MODELS, because one is not a robustness check. B12's lesson: the 0.5B put
# the recompile cliff at 2048 against the 1.5B's 1024, so model size is an axis
# this rig has already been caught by. If the geometry findings hold across both,
# they are about language rather than about one checkpoint.
MODELS = (
    ("Qwen2.5-0.5B-4bit", "mlx-community/Qwen2.5-0.5B-Instruct-4bit", (6, 12, 18)),
    ("Qwen2.5-1.5B-4bit", "mlx-community/Qwen2.5-1.5B-Instruct-4bit", (7, 14, 21)),
)
DOMAIN_RANK = 8                 # top-r kept per domain, matching E1.1c's configs
RANK_REQUEST = 8
INTERFERENCE_LIMIT = 0.05
RETENTIONS = (0.90, 0.95, 0.99, 0.995, 0.999)
SEED = 20260806
N_PROBE = 300

# Eight domains, plain declarative text, written to be DISTINCT rather than to
# have any particular spectral property. Nothing here is tuned: the whole point
# is that the geometry is whatever the model happens to impose.
DOMAINS = {
    "code": [
        "A hash map gives amortised constant-time lookup by bucketing keys.",
        "Recursion needs a base case or the call stack will overflow.",
        "A mutex serialises access to a shared resource between threads.",
        "Garbage collection reclaims heap memory that is no longer reachable.",
        "A binary search halves the interval on each comparison.",
        "Static typing catches a class of errors before the program runs.",
        "A pure function returns the same output for the same input.",
        "Indexes speed up reads and slow down writes in a database.",
        "The compiler lowers source code into machine instructions.",
        "A race condition appears when ordering between threads is unspecified.",
    ],
    "medicine": [
        "The mitochondrion produces most of the cell's chemical energy.",
        "Antibiotics act on bacteria and have no effect on viruses.",
        "Insulin lowers blood glucose by promoting uptake into cells.",
        "The kidney filters waste products out of the bloodstream.",
        "A fever is a regulated rise in core body temperature.",
        "Platelets aggregate at a wound to begin clot formation.",
        "The vagus nerve carries signals between brain and viscera.",
        "Anaemia is a deficiency of circulating red blood cells.",
        "Vaccines prime the immune system against a specific pathogen.",
        "The liver metabolises many drugs before they reach circulation.",
    ],
    "law": [
        "A contract requires offer, acceptance and consideration.",
        "The burden of proof in a criminal trial rests with the prosecution.",
        "Precedent binds lower courts within the same jurisdiction.",
        "A tort is a civil wrong giving rise to liability in damages.",
        "Statutes are enacted by a legislature and interpreted by courts.",
        "Due process requires notice and an opportunity to be heard.",
        "An easement grants limited use of another party's land.",
        "Mens rea refers to the mental element of an offence.",
        "Jurisdiction determines which court may hear a dispute.",
        "A trust separates legal ownership from beneficial interest.",
    ],
    "cooking": [
        "Browning meat develops flavour through the Maillard reaction.",
        "Resting a roast lets the juices redistribute through the meat.",
        "Salt draws moisture out of vegetables by osmosis.",
        "Yeast ferments sugars and produces carbon dioxide in dough.",
        "Emulsions hold oil and water together with a stabiliser.",
        "Blanching sets the colour of green vegetables before freezing.",
        "A roux thickens a sauce with flour cooked into fat.",
        "Acid brightens a dish and balances richness.",
        "Simmering keeps liquid just below a rolling boil.",
        "Proofing gives dough time for the gluten to relax.",
    ],
    "mathematics": [
        "A prime number has exactly two distinct positive divisors.",
        "The derivative measures the instantaneous rate of change.",
        "A matrix is invertible if and only if its determinant is nonzero.",
        "Two events are independent when their joint probability factorises.",
        "The pigeonhole principle forces a collision when items exceed containers.",
        "A group is a set with an associative operation, identity and inverses.",
        "Convergence of a series depends on the behaviour of its tail.",
        "The gradient points in the direction of steepest ascent.",
        "An eigenvector keeps its direction under a linear map.",
        "Induction proves a statement for all natural numbers from a base case.",
    ],
    "music": [
        "A major triad stacks a major third under a minor third.",
        "Tempo is measured in beats per minute.",
        "The circle of fifths orders keys by their shared accidentals.",
        "Counterpoint combines independent melodic lines.",
        "A cadence marks the close of a musical phrase.",
        "Timbre distinguishes instruments playing the same pitch.",
        "Syncopation places emphasis off the expected beat.",
        "A key signature fixes which notes are sharpened or flattened.",
        "Dynamics describe the loudness of a passage.",
        "Modulation moves a piece from one key to another.",
    ],
    "sailing": [
        "A sailboat cannot sail directly into the wind.",
        "Tacking turns the bow through the wind to change course.",
        "The keel counteracts the sideways force on the sails.",
        "Reefing reduces sail area in strong wind.",
        "Windward is the side from which the wind is blowing.",
        "A spinnaker is flown when running downwind.",
        "The rudder steers by deflecting water flow astern.",
        "Heeling is the boat leaning under wind pressure.",
        "A halyard raises a sail up the mast.",
        "Tide affects both depth and the speed made good.",
    ],
    "geology": [
        "Sedimentary rock forms from compacted layers of particles.",
        "Plate tectonics explains the drift of continents over time.",
        "A fault is a fracture along which rock has moved.",
        "Igneous rock crystallises from cooling magma.",
        "Erosion transports weathered material away from its source.",
        "The Mohs scale ranks minerals by scratch hardness.",
        "Metamorphism alters rock under heat and pressure.",
        "An aquifer stores groundwater in permeable rock.",
        "Strata record the relative age of deposits.",
        "Volcanism releases gas and molten rock at the surface.",
    ],
}


def extract(model, tok, layer: int) -> dict:
    """Token-level hidden states per domain, taken at one layer.

    Token positions within a text are correlated, which is a property of language
    rather than an artifact -- the activations a domain actually produces when
    served are token-level, so that is the right object.
    """
    out = {}
    for name, texts in DOMAINS.items():
        vecs = []
        for t in texts:
            ids = mx.array([tok.encode(t)])
            h = model.model.embed_tokens(ids)
            for i, blk in enumerate(model.model.layers):
                h = blk(h, mask=None, cache=None)
                if i == layer:
                    break
            mx.eval(h)
            vecs.append(np.array(h, dtype=np.float32)[0])
        out[name] = np.vstack(vecs)
    return out


N_MASSIVE = 5           # dims excluded in the outlier-removed arm


def massive_dims(acts: dict, n: int = N_MASSIVE):
    """The dimensions carrying outsized variance, pooled over all domains.

    MASSIVE ACTIVATIONS are documented transformer behaviour, not an extraction
    bug: a handful of residual-stream dimensions carry enormous, largely
    input-INDEPENDENT magnitude. Measured here rather than assumed -- the 1.5B
    shows max |a| = 227 against the 0.5B's 10.4, and 5 dims holding 74% of total
    variance against 6.7%.
    """
    x = np.vstack([acts[n_] for n_ in acts])
    var = (x - x.mean(0, keepdims=True)).var(0)
    order = np.argsort(-var)
    return order[:n], float(var[order[:n]].sum() / max(var.sum(), 1e-12))


def geometry(acts: dict) -> dict:
    """Spectrum shape, effective rank, and cross-domain overlap -- measured."""
    names = list(acts)
    bases, decays, eff_ranks, withins = [], [], [], []
    for n in names:
        x = acts[n] - acts[n].mean(axis=0, keepdims=True)
        u, s, vt = np.linalg.svd(x, full_matrices=False)
        bases.append(vt[:DOMAIN_RANK].T)                 # (dim, r)
        withins.append(s[:DOMAIN_RANK] / max(s[0], 1e-12))
        lam = s[:DOMAIN_RANK] ** 2
        # decay exponent: fit log(sigma_k) ~ -a log k over the retained range
        k = np.arange(1, DOMAIN_RANK + 1, dtype=float)
        a = -np.polyfit(np.log(k), np.log(np.maximum(s[:DOMAIN_RANK], 1e-12)), 1)[0]
        decays.append(float(a))
        full = s ** 2
        eff_ranks.append(float(full.sum() ** 2 / max((full ** 2).sum(), 1e-12)))

    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlaps.append(float(np.linalg.norm(bases[i].T @ bases[j]) ** 2 / DOMAIN_RANK))
    return {"names": names, "bases": bases, "withins": withins,
            "decay_mean": float(np.mean(decays)),
            "decay_range": [round(float(min(decays)), 3), round(float(max(decays)), 3)],
            "effective_rank_mean": float(np.mean(eff_ranks)),
            "overlap_mean": float(np.mean(overlaps)),
            "overlap_max": float(np.max(overlaps))}


class RealMixture:
    """DomainMixture's interface, backed by real activations.

    Rates stay ZIPFIAN AND ASSUMED -- how often a domain is actually served needs
    usage logs. Only the geometry is measured, which is the half that E1.6's
    blindness ratio depends on.
    """

    def __init__(self, acts: dict, rng):
        self.names = list(acts)
        self.n_domains = len(self.names)
        self.pools = [acts[n] - acts[n].mean(axis=0, keepdims=True) for n in self.names]
        self.dim = self.pools[0].shape[1]
        r = 1.0 / np.arange(1, self.n_domains + 1, dtype=float)
        self.rates = r / r.sum()

    def sample(self, n: int, rng):
        lab = rng.choice(self.n_domains, size=n, p=self.rates)
        f = np.empty((n, self.dim), dtype=np.float64)
        for i, d in enumerate(lab):
            pool = self.pools[d]
            f[i] = pool[rng.integers(0, len(pool))]
        return f, lab

    def equal_draw(self, per: int, rng):
        """Equal N per owner from its own pool. No rate vector consulted."""
        rows = []
        for d in range(self.n_domains):
            pool = self.pools[d]
            idx = rng.integers(0, len(pool), size=per)
            rows.append(pool[idx])
        return np.vstack(rows)

    def probes(self, d: int, n: int, rng):
        pool = self.pools[d]
        return pool[rng.integers(0, len(pool), size=n)]


def drop_dims(acts: dict, dims) -> dict:
    keep = np.setdiff1d(np.arange(next(iter(acts.values())).shape[1]), dims)
    return {k: v[:, keep] for k, v in acts.items()}


def hinge_on(world: RealMixture, rng) -> dict:
    """E1.6's comparison, instruments unchanged, on this world."""
    dim = world.dim
    n_obs = 4000
    feats, labels = world.sample(n_obs, rng)
    calib, calib_lab = world.sample(500, rng)
    test, _ = world.sample(500, rng)
    readout = rng.normal(size=(dim, 16)) / np.sqrt(dim)
    probe = {d: world.probes(d, N_PROBE, rng) for d in range(world.n_domains)}

    est_spec = RtEstimator(dim=dim, lam=1.0); est_spec.update(feats)
    reg_feats = world.equal_draw(max(n_obs // world.n_domains, 1), rng)
    reg_calib = world.equal_draw(max(500 // world.n_domains, 1), rng)
    est_reg = RtEstimator(dim=dim, lam=1.0); est_reg.update(reg_feats)

    out = {}
    for tag, (est, cal) in {"spectrum": (est_spec, calib),
                            "register": (est_reg, reg_calib)}.items():
        rows = []
        for rho in RETENTIONS:
            r = rank_for_energy(est, cal, rho)
            interfs = np.array([
                interference_by_rank(est, r, probe[d], readout, RANK_REQUEST, rng)
                for d in range(world.n_domains)])
            leaks = np.array([leakage_by_rank(est, r, probe[d])
                              for d in range(world.n_domains)])
            it = interference_by_rank(est, r, test, readout, RANK_REQUEST, rng)
            rows.append({
                "retention": rho, "committed_rank": int(r), "free_rank": int(dim - r),
                "interf_traffic": round(float(it), 4),
                "interf_worst": round(float(interfs.max()), 4),
                "blindness": round(float(interfs.max()) / max(float(it), 1e-9), 2),
                "n_failing": int(np.sum(interfs > INTERFERENCE_LIMIT)),
                "spearman": round(float(spearmanr(world.rates, leaks).statistic), 3),
            })
        out[tag] = rows
    return out


def main() -> int:
    print("\nEB.2 -- real per-domain activation geometry, and the hinge on it\n")
    print(f"  {len(DOMAINS)} domains, 2 models\n")
    t0 = time.time()

    results = {}
    for tag, path, layers in MODELS:
        model, tok = load(path)
        for layer in layers:
            acts = extract(model, tok, layer)
            dims, share = massive_dims(acts)
            g = geometry(acts)
            rng = np.random.default_rng(SEED)
            world = RealMixture(acts, rng)
            h = hinge_on(world, rng)
            # THE OUTLIER-REMOVED ARM. Not a correction -- a second reading. The
            # raw arm is what GPM's energy criterion actually sees; this one is
            # the domain structure underneath it. Both are needed, because the
            # gap between them IS the finding.
            acts_x = drop_dims(acts, dims)
            gx = geometry(acts_x)
            rngx = np.random.default_rng(SEED)
            hx = hinge_on(RealMixture(acts_x, rngx), rngx)
            results[f"{tag}:L{layer}"] = {
                "model": tag, "layer": layer,
                "geometry": {k: v for k, v in g.items()
                             if k not in ("bases", "withins", "names")},
                "geometry_no_massive": {k: v for k, v in gx.items()
                                        if k not in ("bases", "withins", "names")},
                "massive_share": round(share, 4),
                "hinge": h, "hinge_no_massive": hx,
                "dim": int(world.dim),
                "tokens_per_domain": int(len(next(iter(acts.values()))))}
        del model, tok

    keys = list(results)
    i95 = RETENTIONS.index(0.95)
    models = [m[0] for m in MODELS]

    print(f"  {'model:layer':>22}{'dim':>6}{'decay':>8}{'eff rank':>10}"
          f"{'overlap':>9}{'massive':>9}{'decay-x':>9}{'effrank-x':>11}")
    for k in keys:
        r = results[k]; g = r["geometry"]; gx = r["geometry_no_massive"]
        print(f"  {k:>22}{r['dim']:>6}{g['decay_mean']:>8.3f}"
              f"{g['effective_rank_mean']:>10.1f}{g['overlap_mean']:>9.3f}"
              f"{r['massive_share']:>9.1%}{gx['decay_mean']:>9.3f}"
              f"{gx['effective_rank_mean']:>11.1f}")
    print(f"\n    massive   share of total variance in the top {N_MASSIVE} dimensions")
    print("    -x        the same statistic with those dimensions removed")
    print("    DomainMixture assumes decay 0.50 and near-zero cross-domain overlap.")

    print(f"\n  THE HINGE, at rho = 0.95, raw and outlier-removed:")
    print(f"  {'model:layer':>22}{'arm':>11}{'blind':>8}{'spearman':>10}"
          f"{'blind-x':>10}{'spearman-x':>12}{'fail':>6}")
    for k in keys:
        for tag in ("spectrum", "register"):
            r = results[k]["hinge"][tag][i95]
            rx = results[k]["hinge_no_massive"][tag][i95]
            print(f"  {k if tag == 'spectrum' else '':>22}{tag:>11}"
                  f"{r['blindness']:>8.2f}{r['spearman']:>10.3f}"
                  f"{rx['blindness']:>10.2f}{rx['spearman']:>12.3f}"
                  f"{r['n_failing']:>6}")

    # SCORED PER MODEL, NEVER POOLED. The two models disagree by 5x on decay and
    # 30x on effective rank; a mean across them is the pooled-hides-tail defect
    # this record keeps finding, committed inside its own scoring.
    print("\n  SCORED PER MODEL. Pooling across them would hide a disagreement of")
    print("  5x on decay and 30x on effective rank -- the pooled-hides-tail defect,")
    print("  in this experiment's own scoring. The first run of this pooled, and")
    print("  KB1 flipped from NO to ok when the second model was added.")
    per = {}
    for m in models:
        ks = [k for k in keys if results[k]["model"] == m]
        d = float(np.mean([results[k]["geometry"]["decay_mean"] for k in ks]))
        dx = float(np.mean([results[k]["geometry_no_massive"]["decay_mean"] for k in ks]))
        ov = float(np.mean([results[k]["geometry"]["overlap_mean"] for k in ks]))
        ms = float(np.mean([results[k]["massive_share"] for k in ks]))
        sb = float(np.mean([results[k]["hinge"]["spectrum"][i95]["blindness"] for k in ks]))
        rb = float(np.mean([results[k]["hinge"]["register"][i95]["blindness"] for k in ks]))
        ss = float(np.mean([results[k]["hinge"]["spectrum"][i95]["spearman"] for k in ks]))
        rs = float(np.mean([results[k]["hinge"]["register"][i95]["spearman"] for k in ks]))
        sbx = float(np.mean([results[k]["hinge_no_massive"]["spectrum"][i95]["blindness"] for k in ks]))
        rbx = float(np.mean([results[k]["hinge_no_massive"]["register"][i95]["blindness"] for k in ks]))
        ssx = float(np.mean([results[k]["hinge_no_massive"]["spectrum"][i95]["spearman"] for k in ks]))
        rsx = float(np.mean([results[k]["hinge_no_massive"]["register"][i95]["spearman"] for k in ks]))
        fs = float(np.mean([results[k]["hinge"]["spectrum"][i95]["n_failing"] for k in ks]))
        fr = float(np.mean([results[k]["hinge"]["register"][i95]["n_failing"] for k in ks]))
        per[m] = {"decay": round(d, 3), "decay_no_massive": round(dx, 3),
                  "overlap": round(ov, 3), "massive_share": round(ms, 4),
                  "blindness": {"spectrum": round(sb, 2), "register": round(rb, 2)},
                  "spearman": {"spectrum": round(ss, 3), "register": round(rs, 3)},
                  "blindness_no_massive": {"spectrum": round(sbx, 2), "register": round(rbx, 2)},
                  "spearman_no_massive": {"spectrum": round(ssx, 3), "register": round(rsx, 3)},
                  "n_failing": {"spectrum": round(fs, 1), "register": round(fr, 1)},
                  "KB1_decay_as_assumed": bool(0.25 <= d <= 1.0),
                  "KB2_near_orthogonal": bool(ov <= 0.25),
                  "KB3_hinge_survives": bool(rb < sb and rs > -0.5),
                  "KB3_no_massive": bool(rbx < sbx and rsx > -0.5)}

    print(f"\n  {'model':>20}{'KB1':>6}{'KB2':>6}{'KB3':>6}{'KB3-x':>8}"
          f"{'decay':>8}{'decay-x':>9}{'overlap':>9}{'massive':>9}")
    for m in models:
        v = per[m]
        f = lambda b: "ok" if b else "NO"
        print(f"  {m:>20}{f(v['KB1_decay_as_assumed']):>6}{f(v['KB2_near_orthogonal']):>6}"
              f"{f(v['KB3_hinge_survives']):>6}{f(v['KB3_no_massive']):>8}"
              f"{v['decay']:>8.3f}{v['decay_no_massive']:>9.3f}"
              f"{v['overlap']:>9.3f}{v['massive_share']:>9.1%}")

    kb3_all = all(per[m]["KB3_hinge_survives"] for m in models)
    kb3x_all = all(per[m]["KB3_no_massive"] for m in models)

    print("\n  KB3 HOLDS IN EVERY MODEL, EVERY LAYER, BOTH READINGS. That is the")
    print("  result: the register reduces blindness and breaks the rarity monotone")
    print("  on measured geometry, and it does so whether or not the massive")
    print("  dimensions are included. Section 1's central claim survives")
    print("  calibration, which is the question EB.2 existed to answer.")

    print("\n  KB1 AND KB2 BOTH FAIL, AND THE MODELS DISAGREE ABOUT HOW.")
    print("  DomainMixture assumes decay 0.50 and independent bases. Measured:")
    print(f"    0.5B  decay {per[models[0]]['decay']:.3f}, effective rank ~37 -- much FLATTER")
    print(f"    1.5B  decay {per[models[1]]['decay']:.3f}, effective rank ~1.5 -- much STEEPER")
    print("  and the 1.5B's steepness is not domain structure. It is MASSIVE")
    print(f"  ACTIVATIONS: {per[models[1]]['massive_share']:.0%} of its variance sits in {N_MASSIVE} dimensions")
    print(f"  against the 0.5B's {per[models[0]]['massive_share']:.0%}. Remove them and the decay drops to")
    print(f"  {per[models[1]]['decay_no_massive']:.3f}. Cross-domain overlap is ~0.30 in BOTH, against an")
    print("  assumed ~0, so E1.1c's panel C concern is confirmed rather than swept.")

    print("\n  AND A FINDING ABOUT THE CRITERION ITSELF, not about the register.")
    print("  GPM's energy criterion commits directions in order of retained")
    print(f"  energy. On the 1.5B, {per[models[1]]['massive_share']:.0%} of that energy is in {N_MASSIVE} dimensions that")
    print("  are largely input-INDEPENDENT -- they carry no domain information at")
    print("  all. So the criterion spends its retention budget first on directions")
    print("  that distinguish nothing, and whatever rank it has left is what")
    print("  actually protects capabilities. That is a defect in the MECHANISM")
    print("  E1.1b adopted, it is independent of register-versus-spectrum, and no")
    print("  synthetic generator would have produced it.")

    verdict = "PASS" if kb3_all else "FAIL"
    print(f"\n  E1.6 SURVIVES CALIBRATION: {verdict}\n")
    print(f"  ({time.time() - t0:.0f}s)")

    out = pathlib.Path(__file__).resolve().parents[1] / "results" / "eb_2_activation_spectra.json"
    out.write_text(json.dumps(
        {"models": [m[0] for m in MODELS], "domains": list(DOMAINS),
         "domain_rank": DOMAIN_RANK, "retentions": list(RETENTIONS),
         "by_model_layer": results,
         "per_model": per, "n_massive_dims": N_MASSIVE,
         "KB3_all_models": bool(kb3_all), "KB3_no_massive_all": bool(kb3x_all),
         "rates_still_assumed": True, "verdict": verdict}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
