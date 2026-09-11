"""Metrics beyond compiled/correct/genuine.

The benchmark exists to measure whether generated code ENGAGES THE HARDWARE, so
correct scalar code is a task FAILURE, not a success. These are the axes that make
that distinction measurable. Every function here is pure (no I/O, no simulator) and
operates on feedback dicts the evaluator already produces.

Three of these use signal that `evaluate()` has always computed and nobody reported:
`vec_frac`, `pack_p_static`, `static_hvx_insns`.
"""
import re

MECH_FLAG = {"hvx": "used_hvx_compute", "hmx": "used_hmx", "dma": "used_dma",
             "vtcm": "used_vtcm", "l2fetch": "used_l2fetch"}

# Hexagon intrinsics are spelled Q6_<type>_<op>_<operands>. Anything matching this
# shape that the SDK headers do not declare was invented by the model.
_Q6 = re.compile(r"\bQ6_\w+")


def hallucinated_intrinsics(src: str, valid: set) -> set:
    """Q6_ names used in `src` that do not exist in the SDK headers.

    The sharpest axis on which a model trained on a real ISA should beat one that has
    never seen it: a frontier model has no way to know which of ~1392 intrinsic
    spellings are real, and an invented name is a hard compile error. Previously
    measured: 100% of holdout no-compiles traced to exactly this.

    `valid` comes from intrinsic_repair.valid_intrinsics(sdk_root). If it is empty the
    check is not meaningful, so return nothing rather than declare everything fake --
    fail open here, because a false hallucination count would slander a good arm."""
    if not src or not valid:
        return set()
    return {n for n in _Q6.findall(src) if n not in valid}


def hallucination_rate(sources, valid) -> dict:
    """Fraction of generations that name at least one nonexistent intrinsic."""
    srcs = [s for s in sources if s]
    if not srcs or not valid:
        return {"n": len(srcs), "with_hallucination": 0, "rate": None, "distinct": 0}
    bad, names = 0, set()
    for s in srcs:
        h = hallucinated_intrinsics(s, valid)
        if h:
            bad += 1
            names |= h
        # distinct names are reported too: one arm inventing the same name 40 times is a
        # different failure from 40 different inventions
    return {"n": len(srcs), "with_hallucination": bad,
            "rate": bad / len(srcs), "distinct": len(names)}


def declared(spec) -> list:
    return [m for m in MECH_FLAG if m in (spec.get("mechanisms") or [])]


def mechanism_weighted(records) -> dict:
    """Partial credit over declared mechanism SLOTS, not whole tasks.

    tier-3 is all-or-nothing: a kernel that engages HMX and HVX but misses VTCM on a
    3-mechanism task scores identically to one that engages nothing. That hides real
    progress on compound tasks, which are exactly the hard ones. Counted only on
    CORRECT kernels -- engaging hardware in a wrong kernel is not progress."""
    eng = tot = 0
    for rec in records:
        fb, spec = rec["feedback"], rec["spec"]
        for m in declared(spec):
            tot += 1
            if fb.get("correct") and fb.get(MECH_FLAG[m]):
                eng += 1
    return {"engaged": eng, "slots": tot, "rate": (eng / tot if tot else 0.0)}


def conversion(records) -> dict:
    """P(genuine | correct) -- the axis the benchmark is actually about.

    genuine <= correct by construction, so raw genuine counts conflate "can write
    correct code" with "reaches for the accelerator". This isolates the second. A model
    that writes 40 correct scalar kernels scores 0 here, which is the intended verdict."""
    cor = gen = 0
    for rec in records:
        fb, spec = rec["feedback"], rec["spec"]
        if not fb.get("correct"):
            continue
        cor += 1
        if all(fb.get(MECH_FLAG[m]) for m in declared(spec)):
            gen += 1
    return {"correct": cor, "genuine": gen, "rate": (gen / cor if cor else 0.0)}


def isa_quality(records) -> dict:
    """ISA-native code quality over CORRECT kernels: VLIW packing and vector fraction.

    Already computed by evaluate() (pack_p_static, vec_frac) and never reported. Two
    kernels can both be correct-and-genuine while one issues a handful of vector ops in
    poorly packed bundles and the other is densely vectorised."""
    packs = [r["feedback"].get("pack_p_static") for r in records
             if r["feedback"].get("correct") and r["feedback"].get("pack_p_static")]
    vecs = [r["feedback"].get("vec_frac") for r in records
            if r["feedback"].get("correct") and r["feedback"].get("vec_frac") is not None]
    return {"n_packing": len(packs),
            "mean_pack_p_static": (sum(packs) / len(packs) if packs else None),
            "n_vec": len(vecs),
            "mean_vec_frac": (sum(vecs) / len(vecs) if vecs else None)}


def cycles_intersection(recs_a, recs_b) -> dict:
    """Cycle comparison on the tasks BOTH arms got correct -- the only apples-to-apples
    perf view available.

    Comparing mean cycles over each arm's own correct set is meaningless: the arms solve
    different tasks, so the means describe different workloads. Restricting to the
    intersection removes that confound. SIMULATOR-ONLY and uncalibrated to silicon --
    report as directional, never as a hardware speedup."""
    def by_task(recs):
        return {r["spec"].get("task_id"): r["feedback"] for r in recs}
    A, B = by_task(recs_a), by_task(recs_b)
    both, a_faster = [], 0
    for t in sorted(set(A) & set(B)):
        fa, fb_ = A[t], B[t]
        if not (fa.get("correct") and fb_.get("correct")):
            continue
        ca, cb = fa.get("cycles"), fb_.get("cycles")
        if not ca or not cb:
            continue
        both.append((t, ca, cb))
        if ca < cb:
            a_faster += 1
    return {"n_both_correct": len(both), "a_faster": a_faster,
            "b_faster": len(both) - a_faster,
            "mean_ratio_a_over_b": (sum(a / b for _, a, b in both) / len(both)
                                    if both else None),
            "tasks": both}
