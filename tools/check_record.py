"""Consistency check: the record cannot silently drift from the code.

I2 applied to this repo. The whole value of the artifact is that someone else can
check it, and that property decays quietly -- a claim gets a status, an experiment
gets renamed, a constant gets swept and the doc keeps quoting the pinned value.
None of that produces an error until someone tries to rely on it.

Checks, in order of how badly a failure would mislead a reader:

  1. every claim citing a result file has one
  2. every result file is cited by a claim
  3. every experiment has a claim entry
  4. every claim status is from the allowed set
  5. every doc a claim points at exists
  6. constants quoted in the docs match their definitions in code

Run with no arguments. Exits non-zero on any failure, so it can be a CI gate.
"""

from __future__ import annotations

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLAIMS = ROOT / "claims" / "claims.yaml"
RESULTS = ROOT / "results"
VALID_STATUS = {"untested", "running", "PASS", "FAIL", "PARTIAL", "reference"}

# Constants that appear in prose and must match the code that defines them.
# (doc phrase pattern, source file, symbol) -- the value is parsed from source.
QUOTED_CONSTANTS = [
    (r"draw cap of (\d+) entries", "rig_a/experiments/e0_1_verified_recompilability.py", "DRAW_CAP"),
    (r"breadth <= ?([\d.]+)|breadth ≤ ?([\d.]+)", "rig_a/experiments/e0_2e_r9_breadth_coverage.py", "BREADTH_TARGET"),
]


def load_claims() -> list[dict]:
    """Minimal parse -- avoids a pyyaml dependency for a handful of fields."""
    text = CLAIMS.read_text()
    blocks, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^  - id: (\S+)", line)
        if m:
            if cur:
                blocks.append(cur)
            cur = {"id": m.group(1), "_lines": []}
            continue
        if cur is not None:
            cur["_lines"].append(line)
            for field in ("status", "result", "procedure", "tested_against"):
                fm = re.match(rf"^    {field}: (\S+)", line)
                if fm:
                    cur[field] = fm.group(1).strip('"')
    if cur:
        blocks.append(cur)
    return blocks


def main() -> int:
    fails: list[str] = []
    warns: list[str] = []
    claims = [c for c in load_claims() if c["id"].startswith(("E", "B", "R"))]
    experiments = [p for p in (ROOT / "rig_a" / "experiments").glob("e*.py")]
    experiments += [p for p in (ROOT / "rig_b").glob("eb_*.py")]

    # 1 + 5 -- referenced files exist
    for c in claims:
        for field in ("result", "procedure", "tested_against"):
            ref = c.get(field)
            if ref and not (ROOT / ref).exists():
                fails.append(f"{c['id']}: {field} -> missing file {ref}")

    # 2 -- every result file is cited
    cited = {c.get("result") for c in claims if c.get("result")}
    for r in sorted(RESULTS.glob("*.json")):
        rel = f"results/{r.name}"
        if rel not in cited:
            fails.append(f"orphan result: {rel} is cited by no claim")

    # 3 -- every experiment has a claim
    def claim_id_for(path: pathlib.Path) -> str:
        stem = path.stem                       # e0_2e_r9_breadth_coverage
        m = re.match(r"^(e|eb)(\d+)_(\d+)([a-z]?)", stem)
        if not m:
            return ""
        pre = "EB" if m.group(1) == "eb" else "E"
        return f"{pre}{m.group(2)}.{m.group(3)}{m.group(4)}"

    have = {c["id"] for c in claims}
    for p in sorted(experiments):
        cid = claim_id_for(p)
        if cid and cid not in have:
            fails.append(f"{p.relative_to(ROOT)}: no claim entry for {cid}")

    # 4 -- statuses are from the allowed set
    for c in claims:
        st = c.get("status")
        if c["id"].startswith("E") and st and st not in VALID_STATUS:
            fails.append(f"{c['id']}: status '{st}' not in {sorted(VALID_STATUS)}")

    # 6 -- quoted constants match their definitions
    docs = " ".join((ROOT / d).read_text() for d in ("STATUS.md", "PLAN.md")
                    if (ROOT / d).exists())
    for pattern, src, symbol in QUOTED_CONSTANTS:
        srcp = ROOT / src
        if not srcp.exists():
            warns.append(f"constant check: source {src} missing")
            continue
        sm = re.search(rf"^{symbol}\s*=\s*([\d.]+)", srcp.read_text(), re.M)
        if not sm:
            warns.append(f"constant check: {symbol} not found in {src}")
            continue
        defined = float(sm.group(1))
        for hit in re.finditer(pattern, docs):
            quoted = next((g for g in hit.groups() if g), None)
            if quoted and abs(float(quoted) - defined) > 1e-9:
                fails.append(f"{symbol}: docs quote {quoted}, code defines {defined}")

    print(f"record check: {len(claims)} claims, {len(experiments)} experiments, "
          f"{len(list(RESULTS.glob('*.json')))} result files\n")
    for w in warns:
        print(f"  warn  {w}")
    for f in fails:
        print(f"  FAIL  {f}")
    if not fails:
        print("  all checks pass")
    print()
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
