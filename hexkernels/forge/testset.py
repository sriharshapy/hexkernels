"""Coverage-constrained selection over the miner's output.

The miner is mechanical and harvest-ordered, which is what makes the corpus a
statement about the registry rather than about anyone's taste. But harvest
order alone produced a holdout with ONE l2fetch task and no mined HMX at all,
and a per-mechanism count IS the denominator every result is reported against.

So constraints are applied here, on top of selection, and they are declared
rather than emergent. Ops are still chosen in harvest order within each
constraint -- the constraint decides how many of each kind, never which.
"""
import collections
import random

PER_TIER = 80
CORE_PER_TIER = 32
TIERS = ("T0", "T1", "T2", "T3")

# `hexkernels.forge.kernels.HANDWRITTEN_BATCHES` occupies batch numbers 1-15, and
# `kernels.batch(n)` resolves hand-written batches FIRST (see its docstring:
# "Batches 1-15 are hand-written here; 16+ are MINED"). A mined selection
# numbered starting at 1 would silently shadow those 15 hand-written batches
# -- `kernels.batch(1)` would keep returning the hand-written kernel while the
# mined batch 1 became unreachable through the only public accessor, even
# though `all_batches()` still reports the full count. FIRST_BATCH must stay
# above the hand-written range so mined and hand-written batch numbers are
# disjoint.
FIRST_BATCH = 16

# Minimums are counts within the 320. The previous holdout had l2fetch=1,
# which made that column untested rather than failed.
MECHANISM_MINIMUMS = {
    "hvx": 320,
    "l2fetch": 60,
    "vtcm": 60,
    "dma": 60,
    "hmx": 24,
}

# The names the REGISTRY exposes. The public API spellings (`mm`, `addmm`,
# `matmul`, `conv2d`, `softmax`, `layer_norm`) are NOT registry rows -- they
# resolve to these, and asking for them by the public name silently matches
# nothing.
REQUIRED_OPS = {
    "contraction": ("linear", "scaled_dot_product_attention",
                    "_scaled_dot_product_attention_math", "_convolution",
                    "rnn_relu_cell", "rnn_tanh_cell", "kron"),
    "normalisation": ("_safe_softmax", "_log_softmax", "_weight_norm",
                      "renorm", "logsumexp"),
    "pooling": ("avg_pool2d", "max_pool2d", "_adaptive_avg_pool2d",
                "adaptive_max_pool2d", "upsample_bilinear2d",
                "upsample_bicubic2d"),
    "activation": ("gelu", "_prelu_kernel", "elu", "hardswish", "mish",
                   "softplus", "log_sigmoid"),
    "reduction": ("mean", "amax", "amin", "prod", "nansum", "cummin",
                  "logaddexp", "all", "any"),
}


def task_id(spec) -> str:
    """Stable identity: op, overload, dtype, tier -- and the fused epilogue.

    A fused spec carries only its FIRST stage in `op`; the composition lives
    in `stages`. Ignoring it collapses `X then all` and `X then amax` onto one
    id, which drops tasks at selection and makes witness records overwrite
    each other downstream. A single-op spec has no `stages` key at all, and
    keeps the same identity as before: an op at fp32 and at fp16 remain
    different tasks (the HMX task and its negative control).
    """
    ov = spec.get("overload") or "default"
    base = f"{spec['op']}.{ov}.{spec['dtype']}.{spec['tier']}"
    stages = spec.get("stages")
    if stages:
        for stage_op, stage_ov in stages[1:]:
            base += f".{stage_op}.{stage_ov or 'default'}"
    return base


def _take(pool, want, chosen_ids, pred):
    """Harvest-ordered take of up to `want` specs matching `pred`."""
    out = []
    for spec in pool:
        if len(out) >= want:
            break
        tid = task_id(spec)
        if tid in chosen_ids or not pred(spec):
            continue
        chosen_ids.add(tid)
        out.append(spec)
    return out


def select_test_set(pool, per_tier=PER_TIER):
    """320 specs, `per_tier` per size tier, meeting every minimum.

    Raises ValueError naming the shortfall if the pool cannot satisfy a
    constraint. Failing loudly is the point: an under-covered set that reports
    a column of 1 is exactly the defect this replaces.
    """
    chosen_ids = set()
    by_tier = {t: [] for t in TIERS}

    # 1. Required operators first -- they are the reason the set looks like
    #    real workloads rather than only elementwise ops.
    required = sum(REQUIRED_OPS.values(), ())
    for tier in TIERS:
        for op in required:
            if len(by_tier[tier]) >= per_tier:
                break
            by_tier[tier] += _take(
                pool, 1, chosen_ids,
                lambda s, t=tier, o=op: s["tier"] == t and s["op"] == o)

    # 2. Mechanism minimums, scarcest first, so a plentiful mechanism cannot
    #    consume the slots a scarce one needs.
    order = sorted(MECHANISM_MINIMUMS.items(), key=lambda kv: kv[1])
    for mech, minimum in order:
        if mech == "hvx":
            continue  # every task has it; asserted after selection
        have = sum(1 for t in TIERS for s in by_tier[t]
                   if mech in s["mechanisms"])
        for tier in TIERS:
            if have >= minimum:
                break
            room = per_tier - len(by_tier[tier])
            if room <= 0:
                continue
            got = _take(pool, min(room, minimum - have), chosen_ids,
                        lambda s, t=tier, m=mech: (s["tier"] == t
                                                   and m in s["mechanisms"]))
            by_tier[tier] += got
            have += len(got)
        if have < minimum:
            raise ValueError(
                f"pool cannot meet the minimum for {mech}: {have} < {minimum}")

    # 3. Fill the rest in harvest order.
    for tier in TIERS:
        room = per_tier - len(by_tier[tier])
        by_tier[tier] += _take(pool, room, chosen_ids,
                               lambda s, t=tier: s["tier"] == t)
        if len(by_tier[tier]) < per_tier:
            raise ValueError(
                f"pool cannot fill {tier}: {len(by_tier[tier])} < {per_tier}")

    out = [s for t in TIERS for s in by_tier[t]]
    for s in out:
        if not s["mechanisms"]:
            raise ValueError(f"{task_id(s)} has no mechanisms; all([]) is True")
    return out


def designate_core(test_set, per_tier=CORE_PER_TIER, seed=0):
    """`per_tier` task ids per size tier, chosen deterministically.

    HMX-bearing tasks are taken first within each tier so the scarcest column
    keeps a usable denominator in the core; the remainder is a seeded sample.
    """
    rng = random.Random(seed)
    core = []
    for tier in TIERS:
        rows = [s for s in test_set if s["tier"] == tier]
        hmx = [s for s in rows if "hmx" in s["mechanisms"]]
        rest = [s for s in rows if "hmx" not in s["mechanisms"]]
        rng.shuffle(rest)
        picked = (hmx + rest)[:per_tier]
        if len(picked) < per_tier:
            raise ValueError(f"{tier} has {len(picked)} < {per_tier} tasks")
        core += [task_id(s) for s in picked]
    return core


def core_strata(test_set, core_ids):
    """Sampling weight per core task, so aggregates can be corrected.

    The core oversamples HMX on purpose: proportional sampling would leave a
    denominator too small to report against. Oversampling a scarce stratum is
    sound only if the weights survive to the analysis, so they are recorded
    here rather than reconstructed later from a description of the method.

    weight = (stratum's share of the full set) / (its share of the core).
    A weight of 1.0 means the stratum is represented at its true rate; the
    HMX stratum will come back well below 1.0, and that number is the
    correction factor for any core-level aggregate.

    Computed WITHIN tier -- the core is `per_tier` per tier by construction,
    so tier is already balanced and the weight must not double-correct for
    it.
    """
    core_id_set = set(core_ids)
    strata = {}
    for tier in TIERS:
        rows = [s for s in test_set if s["tier"] == tier]
        total = len(rows)
        if total == 0:
            continue
        full_share = {}
        for stratum in ("hmx", "non_hmx"):
            n = sum(1 for s in rows if ("hmx" in s["mechanisms"]) ==
                    (stratum == "hmx"))
            full_share[stratum] = n / total

        core_rows = [s for s in rows if task_id(s) in core_id_set]
        core_total = len(core_rows)
        core_share = {}
        for stratum in ("hmx", "non_hmx"):
            n = sum(1 for s in core_rows if ("hmx" in s["mechanisms"]) ==
                    (stratum == "hmx"))
            core_share[stratum] = (n / core_total) if core_total else 0.0

        for s in core_rows:
            stratum = "hmx" if "hmx" in s["mechanisms"] else "non_hmx"
            sc = core_share[stratum]
            sf = full_share[stratum]
            weight = (sf / sc) if sc else 0.0
            strata[task_id(s)] = {"stratum": stratum, "tier": tier,
                                   "weight": weight}
    return strata


def _write(path, payload):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None):
    """Generate benchmark/selection.json and benchmark/eval_core.json.

    Usage: python -m hexkernels.forge.testset --pool <path-to-pool.json>

    The pool is the miner's output. It is passed in rather than mined here so
    that regenerating the artifacts is reproducible from a committed file and
    does not depend on the installed torch build's registry drifting. The
    pool is the MINED pool only -- hexkernels.forge.kernels.HANDWRITTEN_BATCHES is
    never imported here, because the mined pool's mechanical provenance is
    the corpus's central claim and the hand-written kernels are a separate,
    disclosed supplement handled outside this module.
    """
    import argparse
    import json
    import pathlib

    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    with open(args.pool, encoding="utf-8") as f:
        raw = json.load(f)
    pool = (raw if isinstance(raw, list)
            else [k for grp in raw["batches"] for k in grp])

    test_set = select_test_set(pool)
    core = designate_core(test_set, seed=args.seed)
    strata = core_strata(test_set, core)

    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path(__file__).resolve().parents[1] / "benchmark")
    # TWO views of the same 320. `specs` is the flat list the audit and tests
    # read; `batches` is what hexkernels.forge.mined._load requires (it KeyErrors on
    # anything else) and is how the witness build reaches these tasks.
    per_batch = 5
    batches = [test_set[i:i + per_batch]
               for i in range(0, len(test_set), per_batch)]
    _write(out / "selection.json",
           {"n": len(test_set), "per_tier": PER_TIER,
            "first_batch": FIRST_BATCH,
            "per_batch": per_batch, "specs": test_set, "batches": batches})
    _write(out / "eval_core.json",
           {"n": len(core), "per_tier": CORE_PER_TIER, "seed": args.seed,
            "task_ids": core, "strata": strata})
    print(f"wrote {len(test_set)} specs and a {len(core)}-task core to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


