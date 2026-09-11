"""Set-level audit. Reports what cannot be checked one task at a time.

Three properties live at the set level. Mechanism counts are the denominators
every result is reported against. Depth-vs-tier balance decides whether "big
is hard" can be told apart from "deep is hard". And the fp32 negative controls
are what make "we grant HMX correctly" a claim rather than a tautology.

`violations` is empty or the set is not ready to freeze.

Beyond the three checked properties, the report also DISCLOSES three things
that are true of the set but are not violations of anything: the single-op /
fused composition (the fused share was deliberately raised to reach 80 tasks
per tier), which required operators are absent and why, and the fact that
mining against this repo is incremental rather than idempotent. An audit that
silently omits known gaps is hiding them, which is the exact failure the
audit exists to prevent.
"""
import collections

from hexkernels.forge import testset

MIN_DEPTH_SPREAD = 2  # distinct n_primitives values required within a tier

# Required operators known to be absent from benchmark/selection.json, and
# why. Two are correct exclusions; one is a real defect that a re-mine would
# fix. This dict is disclosure, not enforcement: it never produces a
# violation, and it is cross-checked against the live specs each run so a
# stale entry (the op has reappeared) is visible rather than silently wrong.
KNOWN_ABSENT = {
    "amax": ("correctly excluded -- already in the corpus as dedicated "
             "hand-written kernels (kernels.py:573 fp32_amax_all, :1480 "
             "fp32_amax_hw); mine.py:770 rejects it as already-in-corpus, "
             "so re-mining it would duplicate an existing task."),
    "amin": ("correctly excluded by the letter of the same rule, and worth "
             "flagging as arguable -- amin is 'used' only as half of "
             "fp32_aminmax (kernels.py:734), which traces to exactly two "
             "nodes. The dedup rule treats 'appears anywhere in an existing "
             "kernel's decomposition' the same as 'was mined as its own "
             "task', so no standalone amin task can ever exist."),
    "gelu": ("a real defect, not a correct exclusion -- nothing rejects it: "
             "it is mechanism-eligible, absent from corpus_targets() (the "
             "existing fp16_gelu is CompositeImplicit and expands before any "
             "aten.gelu node exists), sizes at all four tiers with hvx "
             "granted, and collides with no signature. It is missing only "
             "because a mining run to recover it could not be completed in "
             "this environment. frac is absent with the same symptom, "
             "though frac is not a REQUIRED_OPS entry."),
}

MINING_NOTE = (
    "kernels.all_batches() is HANDWRITTEN_BATCHES union mined.MINED_BATCHES, "
    "and hexkernels/forge/mined.py builds MINED_BATCHES from benchmark/selection.json. "
    "corpus_targets() and corpus_signatures() both walk it -- so "
    "selection.json feeds back into the miner's own exclusion set. Once the "
    "selection exists, a re-mine skips everything already in it and yields "
    "only what is new: the pool is the union of successive mining runs under "
    "a growing exclusion set, and reproducing from scratch would give a "
    "different, larger first run. That is defensible, but it must be stated "
    "rather than discovered."
)

# n_plumbing is 0 on every spec BY CONSTRUCTION: _would_be_destroyed
# (mine.py:191) rejects any candidate containing a non-primitive_admissible
# node before it can be picked, using the same admissibility test
# _depth_counts reuses. It is a structural constant, not a signal, so it is
# never turned into a statistic here -- see the note this string is used in.
N_PLUMBING_NOTE = (
    "n_plumbing is 0 on every spec by construction (mine.py:191 "
    "_would_be_destroyed rejects any candidate containing a non-"
    "primitive_admissible node before it can be picked). It carries no "
    "information and is not reported as a statistic here."
)


def _is_fused(spec) -> bool:
    """A fused spec is identified by a `stages` key; single-op specs lack it.
    Never inferred from the schema string."""
    return bool(spec.get("stages"))


def _composition(specs):
    by_tier = {t: {"single": 0, "fused": 0} for t in testset.TIERS}
    by_mech = collections.defaultdict(lambda: {"single": 0, "fused": 0})
    for s in specs:
        kind = "fused" if _is_fused(s) else "single"
        by_tier.setdefault(s["tier"], {"single": 0, "fused": 0})
        by_tier[s["tier"]][kind] += 1
        for m in s["mechanisms"]:
            by_mech[m][kind] += 1
    return {"by_tier": by_tier, "by_mechanism": dict(by_mech)}


def _known_absent(specs):
    present = {s["op"] for s in specs}
    out = {}
    for op, reason in KNOWN_ABSENT.items():
        out[op] = {"absent": op not in present, "reason": reason}
    return out


def audit(specs, core_ids):
    tier_counts = collections.Counter(s["tier"] for s in specs)

    mech_counts = collections.Counter()
    for s in specs:
        for m in s["mechanisms"]:
            mech_counts[m] += 1

    depth_by_tier = {}
    for tier in testset.TIERS:
        rows = [s for s in specs if s["tier"] == tier]
        depth_by_tier[tier] = dict(
            sorted(collections.Counter(
                s.get("n_primitives") for s in rows).items(),
                key=lambda kv: (kv[0] is None, kv[0])))

    contractions = set(testset.REQUIRED_OPS["contraction"])
    controls = [s for s in specs
                if s["op"] in contractions and s["dtype"] == "float32"]

    v = []

    for mech, minimum in testset.MECHANISM_MINIMUMS.items():
        if mech == "hvx":
            if mech_counts[mech] != len(specs):
                v.append(f"hvx is on {mech_counts[mech]} of {len(specs)} tasks;"
                         " every task must be entitled to hvx")
            continue
        if mech_counts[mech] < minimum:
            v.append(f"{mech}: {mech_counts[mech]} tasks < minimum {minimum}")

    for s in specs:
        if not s["mechanisms"]:
            v.append(f"{testset.task_id(s)} has no mechanisms "
                     "(all([]) is True, so it would score genuine vacuously)")

    if not controls:
        v.append("no fp32 contraction negative control in the set; the HMX "
                 "entitlement rule is then untested on the refusing side")
    for s in controls:
        if "hmx" in s["mechanisms"]:
            v.append(f"{testset.task_id(s)} is fp32 yet granted hmx; "
                     "HMX_MAX_DTYPE_BYTES is 2")

    for tier, dist in depth_by_tier.items():
        known = {k: n for k, n in dist.items() if k is not None}
        if known and len(known) < MIN_DEPTH_SPREAD:
            v.append(f"{tier} depth is degenerate ({known}); big-is-hard "
                     "cannot be separated from deep-is-hard")

    return {"n": len(specs), "n_core": len(core_ids),
            "tier_counts": dict(sorted(tier_counts.items())),
            "mechanism_counts": dict(mech_counts.most_common()),
            "depth_by_tier": depth_by_tier,
            "negative_controls": [testset.task_id(s) for s in controls],
            "composition": _composition(specs),
            "known_absent": _known_absent(specs),
            "mining_note": MINING_NOTE,
            "n_plumbing_note": N_PLUMBING_NOTE,
            "violations": v}


def render(report) -> str:
    L = [f"# Test set audit", "",
         f"{report['n']} tasks; {report['n_core']}-task evaluation core.", "",
         "## Tier counts", ""]
    for t, n in report["tier_counts"].items():
        L.append(f"- {t}: {n}")
    L += ["", "## Mechanism counts", "",
          "| mechanism | tasks | minimum |", "|---|---|---|"]
    for m, n in report["mechanism_counts"].items():
        L.append(f"| {m} | {n} | {testset.MECHANISM_MINIMUMS.get(m, '-')} |")
    by_tier = report["composition"]["by_tier"]
    n_single = sum(c["single"] for c in by_tier.values())
    n_fused = sum(c["fused"] for c in by_tier.values())
    n_total = n_single + n_fused
    fused_pct = (100.0 * n_fused / n_total) if n_total else 0.0
    L += ["", "## Composition (single-op vs fused)", "",
          f"This selection's fused share, computed from the data below, is "
          f"{fused_pct:.0f}% ({n_fused} of {n_total}). That is a different "
          "measurement from the pool-level figure the fused top-up decision "
          "was made against: the mined pool's fused share was deliberately "
          "raised from ~34% to ~44% so the selection could reach its "
          "per-tier quota once the mechanism minimums were also satisfied. "
          "(That 34%/44% pair is historical -- the share of the pool at the "
          "time the top-up decision was made -- and is not derivable from "
          "today's pool or selection; do not mistake it for a computed "
          "figure.) A reader must be able to see the selection's own "
          "fused share and discount it. `fused` is derived from the "
          "presence of a `stages` key, never from the schema string.", "",
          "By tier:", "",
          "| tier | single-op | fused |", "|---|---|---|"]
    for t in testset.TIERS:
        c = report["composition"]["by_tier"].get(t, {"single": 0, "fused": 0})
        L.append(f"| {t} | {c['single']} | {c['fused']} |")
    L += ["", "By mechanism:", "",
          "| mechanism | single-op | fused |", "|---|---|---|"]
    for m, c in report["composition"]["by_mechanism"].items():
        L.append(f"| {m} | {c['single']} | {c['fused']} |")
    L += ["", "## Depth (n_primitives) by tier", ""]
    for t, dist in report["depth_by_tier"].items():
        L.append(f"- {t}: {dist}")
    L += ["", f"Note: {report['n_plumbing_note']}"]
    L += ["", "## Negative controls (fp32 contractions, hmx must be absent)", ""]
    L.append(f"{len(report['negative_controls'])}: "
             f"{', '.join(report['negative_controls']) or 'NONE'}")
    L += ["", "## Known-absent required operators", "",
          "Not the same kind of absence -- two are correct exclusions and "
          "one is a defect. See `hexkernels.forge.audit.KNOWN_ABSENT`.", ""]
    for op, entry in report["known_absent"].items():
        status = "absent" if entry["absent"] else \
            "PRESENT -- stale waiver, this entry no longer applies"
        L.append(f"- `{op}` ({status}): {entry['reason']}")
    L += ["", "## Methodological note: mining is incremental, not idempotent",
          "", report["mining_note"]]
    L += ["", "## Violations", ""]
    L.append("None — the set is ready to freeze."
             if not report["violations"]
             else "\n".join(f"- {x}" for x in report["violations"]))
    return "\n".join(L) + "\n"


def main(argv=None):
    import argparse
    import json
    import pathlib

    ap = argparse.ArgumentParser()
    bench = pathlib.Path(__file__).resolve().parents[1] / "benchmark"
    ap.add_argument("--selection", default=str(bench / "selection.json"))
    ap.add_argument("--core", default=str(bench / "eval_core.json"))
    ap.add_argument("--out", default=str(bench / "AUDIT.md"))
    args = ap.parse_args(argv)

    with open(args.selection, encoding="utf-8") as f:
        specs = json.load(f)["specs"]
    with open(args.core, encoding="utf-8") as f:
        core = json.load(f)["task_ids"]

    rep = audit(specs, core)
    text = render(rep)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    print(text)
    return 1 if rep["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
