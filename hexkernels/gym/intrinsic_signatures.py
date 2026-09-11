"""Signature-aware grounding: put the RIGHT real intrinsics in front of the model.

Constrained decoding (:mod:`intrinsic_grammar`) proved a free *safety* guarantee -- zero
fabricated ``Q6_`` names -- but correctness stayed FLAT: forbidding a fake name does not
teach the model the real one. Both failure modes (fabricated name, and wrong-but-real name)
share one root cause: the model knows the HVX naming *grammar*, not the *lexicon*.

This module attacks the lexicon directly and on-distribution: parse the SDK header's
``C Intrinsic Prototype:`` / ``Assembly Syntax:`` blocks into a signature index, retrieve the
handful relevant to a task (dtype lane + op synonyms), and inject them into the prompt. No
decode perturbation -> no CPU mask bottleneck, any GPU is fine.

Leak-guard: signatures come from the PUBLIC SDK header only -- never from a holdout
``expert.c``. Showing the header is what any Hexagon developer has; showing the answer kernel
would be leaking.

Plan: docs/superpowers/plans/2026-07-30-signature-grounding-plan.md
"""
import os
import re

from hexkernels.gym.intrinsic_repair import _header_paths

#: comment block: /* ==== ... ==== */ wrapping the Assembly Syntax + C Intrinsic Prototype lines
_BLOCK = re.compile(r"/\*\s*=+(.*?)=+\s*\*/", re.S)
_ASM = re.compile(r"Assembly Syntax:\s*(.+)")
_PROTO = re.compile(r"C Intrinsic Prototype:\s*(.+)")
_NAME = re.compile(r"\bQ6_[A-Za-z0-9_]+\b")

_CACHE = {}


def signature_index(sdk_root):
    """``{name: {"proto": str, "asm": str}}`` for every intrinsic declared in the SDK headers.

    Parsed from the generated doc-comment above each ``#define`` (both the HVX and the scalar
    protos headers), so it is exactly as authoritative as the header itself. Cached per SDK."""
    key = os.path.abspath(sdk_root)
    if key in _CACHE:
        return _CACHE[key]
    index = {}
    for h in _header_paths(sdk_root):
        try:
            with open(h, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        for m in _BLOCK.finditer(text):
            body = m.group(1)
            proto = _PROTO.search(body)
            if not proto:
                continue
            proto = proto.group(1).strip()
            name = _NAME.search(proto)
            if not name:
                continue
            asm = _ASM.search(body)
            index[name.group()] = {"proto": proto, "asm": asm.group(1).strip() if asm else ""}
    _CACHE[key] = index
    return index


def render_signatures(names, index):
    """A compact one-line-per-intrinsic block: verbatim C prototype + the asm form as a comment.

    Verbatim on purpose -- the parameter names (``Vu``/``Vv``) line up with the asm operands,
    and rewriting the header risks teaching a signature the compiler will reject. Unknown
    names are dropped (fail-quiet: grounding must never inject a name we cannot vouch for)."""
    lines = []
    for n in names:
        e = index.get(n)
        if not e:
            continue
        lines.append("  %s%s" % (e["proto"], ("   // " + e["asm"]) if e["asm"] else ""))
    return "\n".join(lines)


# ---- retrieval -------------------------------------------------------------------------

#: dtype token -> HVX lane suffixes (the ``.b`` / ``.hf`` fields in the asm syntax).
_DTYPE_LANES = [
    ("uint8", ("ub",)), ("u8", ("ub",)),
    ("int8", ("b",)), ("i8", ("b",)), ("sint8", ("b",)),
    ("uint16", ("uh",)), ("u16", ("uh",)),
    ("int16", ("h",)), ("i16", ("h",)),
    ("uint32", ("uw",)), ("u32", ("uw",)),
    ("int32", ("w",)), ("i32", ("w",)),
    ("int64", ("w",)), ("i64", ("w",)),
    ("fp16", ("hf", "qf16")), ("f16", ("hf", "qf16")), ("half", ("hf", "qf16")),
    ("fp32", ("sf", "qf32")), ("f32", ("sf", "qf32")), ("float", ("sf", "qf32")),
]

#: NN/kernel op words in the prompt -> intrinsic op tokens to prefer.
#:
#: Deliberately EXCLUDES words that collide with ordinary prose in these prompts -- "and",
#: "or", "not", "where", "count", "table", "sub" ("sub-tile remainder"!), "scale". Measured on
#: real task prompts, those pulled whole families of irrelevant intrinsics (vand predicate ops
#: into a ReLU task) and crowded out the right answer. Unambiguous spellings are kept instead.
_OP_SYNONYMS = {
    "relu": ("vmax",), "clamp": ("vmax", "vmin"), "clip": ("vmax", "vmin"),
    "max": ("vmax",), "min": ("vmin",), "maximum": ("vmax",), "minimum": ("vmin",),
    "add": ("vadd",), "sum": ("vadd", "vrmpy"), "accumulate": ("vadd", "vrmpy"),
    "subtract": ("vsub",), "minus": ("vsub",), "difference": ("vsub",),
    "mul": ("vmpy",), "multiply": ("vmpy",), "product": ("vmpy",),
    "matmul": ("vrmpy", "vdmpy", "vmpy"), "gemm": ("vrmpy", "vdmpy", "vmpy"),
    "gemv": ("vrmpy", "vdmpy"), "dot": ("vrmpy", "vdmpy"), "mac": ("vmpy", "vrmpy"),
    "conv": ("vrmpy", "vdmpy", "vmpy"), "reduce": ("vadd", "vrmpy"),
    "shift": ("vasr", "vlsr", "vasl"), "rshift": ("vasr", "vlsr"),
    "requant": ("vasr", "vmpy", "vpack"), "quantize": ("vasr", "vmpy", "vpack"),
    "dequant": ("vmpy", "vsub"),
    "pack": ("vpack", "vshuff"), "unpack": ("vunpack", "vshuff"),
    "shuffle": ("vshuff",), "deal": ("vdeal",), "transpose": ("vshuff", "vdeal"),
    "splat": ("vsplat",), "broadcast": ("vsplat",), "zero": ("vzero",),
    "mux": ("vmux",), "select": ("vmux",), "blend": ("vmux",),
    "compare": ("vcmp",), "greater": ("vcmp",), "equal": ("vcmp",),
    "abs": ("vabs",), "average": ("vavg",), "mean": ("vavg", "vadd"),
    "round": ("vround",), "saturate": ("vsat",), "saturating": ("vsat",),
    "narrow": ("vpack", "vsat"), "widen": ("vunpack", "vsxt", "vzxt"),
    "bitwise": ("vand", "vor", "vxor"), "xor": ("vxor",),
    "swap": ("vswap",), "align": ("valign", "vlalign"), "rotate": ("vror",),
    "convert": ("vconv", "vcvt"),
    "softmax": ("vmax", "vsub", "vmpy"), "exp": ("vmpy",), "sigmoid": ("vmpy",),
    "argmax": ("vmax", "vcmp"), "histogram": ("vhist",),
    "lookup": ("vlut",), "lut": ("vlut",), "gather": ("vgather",), "scatter": ("vscatter",),
    # structural / plumbing words that DO show up in these prompts
    "interleave": ("vshuff", "vdeal"), "deinterleave": ("vdeal", "vshuff"),
    "remainder": ("vsetq",), "tail": ("vsetq",), "predicate": ("vsetq", "vmux"),
    "unaligned": ("valign",), "stride": ("vdeal", "vshuff"),
    "nhwc": ("vshuff", "vdeal"), "nchw": ("vshuff", "vdeal"), "layout": ("vshuff", "vdeal"),
    "reverse": ("vdeal", "vror"), "shuff": ("vshuff",),
    "sqrt": ("vrsqrt",), "rsqrt": ("vrsqrt",), "reciprocal": ("vrecip",), "divide": ("vrecip",),
    "norm": ("vrmpy", "vadd"), "l2": ("vrmpy", "vadd"), "square": ("vrmpy", "vmpy"),
}

#: narrow lane -> the wider lanes a kernel on that dtype accumulates into. An int8 kernel is
#: almost never pure-int8 internally: vmpy widens b*b->h and vrmpy b*b->w, so the w/h ops are
#: exactly what the expert kernels use. Filtering them out (measured) cost ~half of retrieval
#: recall; they are admitted but scored below an exact lane match.
_ACCUM_LANES = {
    "b": ("h", "w"), "ub": ("uh", "uw", "h", "w"),
    "h": ("w",), "uh": ("uw", "w"),
    "hf": ("qf16", "sf", "qf32"), "qf16": ("hf", "sf", "qf32"),
    "sf": ("qf32",), "qf32": ("sf",),
}

#: structural "plumbing" every non-trivial HVX kernel needs regardless of its arithmetic op:
#: split a widened pair (hi/lo), build a tail predicate for the remainder (vsetq), realign an
#: unaligned window (valign), rejoin two vectors (vcombine), compare/select, and the
#: widen/narrow conversions that every int8<->int32 pipeline runs through. Measured: these were
#: the single biggest block of retrieval misses, and no op-synonym in the prompt ever names them.
_PLUMBING_OPS = ("hi", "lo", "vsetq", "valign", "vcombine", "vcmp",
                 "vsxt", "vzxt", "vunpack", "vpack", "vpacke", "vshuff", "vdeal", "vror")

#: ops the SAME kernel legitimately needs at several widths -- an int8 dot product adds in w,
#: shifts in w, then packs back to b. The kit therefore offers these at the exact lane AND at
#: each accumulator lane; the remaining misses after the first recall fix were almost entirely
#: the wide-lane variants of these (Q6_Vw_vadd_VwVw for an int8 task, Q6_Vh_vsplat_R, ...).
_LADDER_OPS = ("vadd", "vsub", "vmpy", "vmpyi", "vasr", "vsplat", "vmax", "vmin")

#: mechanism (spec['mechanisms'] / prompt word) -> (essential ops, nice-to-have ops).
#: The essentials are guaranteed first: a DMA task cannot be solved without dmstart/dmwait, and
#: genuine-mechanism use is the metric we care most about. The secondary ones (dmpause/dmpoll/
#: dmlink/dmresume) are only worth slots once the building blocks are in.
_MECHANISM_OPS = {
    "dma": (("dmstart", "dmwait"), ("dmpoll", "dmpause", "dmlink", "dmresume")),
    "l2fetch": (("l2fetch",), ()),
    "gather": (("vgather",), ()),
    "scatter": (("vscatter",), ()),
}

#: always-offered building blocks (the "base kit"), best lane-match picked per dtype.
_BASE_OPS = ("vadd", "vsub", "vmax", "vmin", "vmpy", "vzero", "vsplat", "vmux")

_LANE_IN_ASM = re.compile(r"\.(qf32|qf16|ub|uh|uw|hf|sf|b|h|w)\b")
_WORDS = re.compile(r"[a-z0-9]+")


def dtype_lanes(dtype):
    """Lane suffixes implied by a spec dtype string.

    Handles the compound forms in the benchmark (``'uint8xint8->int32'``,
    ``'int8->int32->int8'``): every dtype token mentioned contributes its lanes, because a
    mixed-precision kernel legitimately needs intrinsics on both the input and output lane.
    Longest-token-first so ``uint8`` is not shadowed by ``int8``."""
    s = (dtype or "").lower()
    lanes = set()
    for tok, ls in _DTYPE_LANES:
        if tok in s:
            lanes.update(ls)
            s = s.replace(tok, " ")     # consume so int8 doesn't re-match inside uint8
    return lanes


def _op_token(name):
    """The op field of a ``Q6_...`` name (``Q6_Vb_vmax_VbVb`` -> ``vmax``).

    The op is the first lowercase-initial field: type/lane fields are capitalised (``Vb``,
    ``W``, ``R``, ``P``), so this also handles the names that have NO leading type field --
    ``Q6_dmstart_A`` -> ``dmstart``, ``Q6_l2fetch_AP`` -> ``l2fetch`` -- which a fixed
    ``parts[2]`` would misread as the argument field."""
    for p in name.split("_")[1:]:
        if p and p[0].islower():
            return p
    return ""


def _entry_lanes(entry):
    """Lane suffixes the intrinsic operates on, read off its asm syntax (empty = lane-agnostic,
    e.g. ``Vd32=#0``)."""
    return set(_LANE_IN_ASM.findall(entry.get("asm") or ""))


def wanted_ops(prompt):
    """Intrinsic op tokens implied by the words in a task prompt (via :data:`_OP_SYNONYMS`)."""
    ops = set()
    for w in _WORDS.findall((prompt or "").lower()):
        ops.update(_OP_SYNONYMS.get(w, ()))
    return ops


def accumulator_lanes(lanes):
    """The wider lanes a kernel on ``lanes`` naturally accumulates into (see :data:`_ACCUM_LANES`)."""
    out = set()
    for l in lanes:
        out.update(_ACCUM_LANES.get(l, ()))
    return out - set(lanes)


def _pick_canonical(op, lanes, index):
    """The canonical intrinsic for an op at these lanes: the SHORTEST matching name, i.e. the
    plain form (``Q6_Vb_vadd_VbVb``) rather than a ``_sat``/``_rnd``/``_acc`` variant. HVX is
    preferred over the scalar-GPR namesake (``Q6_P_vaddb_PP``). None if the op has no match."""
    ok = set(lanes) | accumulator_lanes(lanes)
    exact = []
    wide = []
    for n, e in index.items():
        if _op_token(n) != op:
            continue
        el = _entry_lanes(e)
        if el and lanes and not (el & ok):
            continue                          # wrong lane even allowing accumulators
        (exact if (not el or not lanes or (el & set(lanes))) else wide).append(n)
    cands = exact or wide
    if not cands:
        return None
    return min(cands, key=lambda n: (0 if "HVX_" in index[n]["proto"] else 1, len(n), n))


def _required_kit(prompt, mechanisms, lanes, index):
    """Intrinsics this task cannot be solved without: the dtype's base building blocks (offered
    across the widening ladder), the structural plumbing (pair halves, tail predicate, align,
    combine, compare, widen/narrow) and every mechanism the spec targets (a DMA task needs
    dmstart/dmwait). Guaranteed into the retrieved set even when relevance scoring would rank
    them below the cut."""
    text = (prompt or "").lower()
    kit = []

    def add(n):
        if n and n not in kit:
            kit.append(n)

    # Priority order matters: the kit is truncated to k, so the most task-critical entries go
    # first. A DMA task that loses dmstart to a wide-lane vadd variant cannot be solved at all.
    active = [tiers for mech, tiers in _MECHANISM_OPS.items()
              if mech in (mechanisms or ()) or mech in text]
    for op in [o for essential, _ in active for o in essential]:     # 1. mechanism-essential
        add(_pick_canonical(op, lanes, index))
    for op in _BASE_OPS:                      # 2. base building blocks at the exact dtype lane
        add(_pick_canonical(op, lanes, index))
    for op in _PLUMBING_OPS:                  # 3. structural plumbing
        add(_pick_canonical(op, lanes, index))
    for op in [o for _, extra in active for o in extra]:             # 4. mechanism-secondary
        add(_pick_canonical(op, lanes, index))
    for lane in sorted(accumulator_lanes(lanes)):                    # 5. the widening ladder
        for op in _LADDER_OPS:
            add(_pick_canonical(op, {lane}, index))
    return kit


def retrieve_for_task(prompt, dtype, index, k=32, mechanisms=None):
    """Up to ``k`` real intrinsic names relevant to a task, best first.

    Heuristic (deliberately simple, tuned by the A/B): score every indexed intrinsic by
    op-synonym match against the prompt + lane match against the spec dtype, then RESERVE slots
    for the required kit (dtype base ops + the spec's target mechanisms) so the essentials
    survive the top-k cut. Returns names only; :func:`render_signatures` renders the block."""
    lanes = dtype_lanes(dtype)
    accum = accumulator_lanes(lanes)
    ops = wanted_ops(prompt)

    scored = []
    for n, e in index.items():
        op = _op_token(n)
        el = _entry_lanes(e)
        score = 0
        if op in ops:
            score += 4
        elif any(o in op for o in ops):
            score += 2                        # vmpyi/vmpye under a 'vmpy' request
        if el and lanes:
            if el & lanes:
                score += 3                    # exact dtype lane
            elif el & accum:
                score += 1                    # the widened lane this dtype accumulates into
            else:
                continue                      # wrong lane: never offer it
        elif not el:
            score += 1                        # lane-agnostic (vzero, vsplat, dmstart)
        if op in _BASE_OPS or op in _PLUMBING_OPS:
            score += 1
        if "HVX_" in e["proto"]:
            score += 1                        # prefer HVX over the scalar-GPR namesake
        if score:
            scored.append((-score, n))
    scored.sort()
    ranked = [n for _, n in scored]

    # NOTE (measured, holdout recall vs the expert kernels): reserving slots for the relevance
    # ranking by capping the kit HURTS at every k (60.2 % vs 75.2 % at k=56) -- the kit, not the
    # prompt-word ranking, is what actually covers the expert lexicon. So the kit takes what it
    # needs and relevance fills the remainder. The corollary is that this retriever wants a
    # generous k: recall is 35 % at k=24 but 75 % at k=56 (see RESULTS_grounding.md).
    kit = _required_kit(prompt, mechanisms, lanes, index)[:k]
    # reserve the kit's slots, fill the rest by relevance, then emit in relevance order
    rest = [n for n in ranked if n not in kit][:max(0, k - len(kit))]
    order = {n: i for i, n in enumerate(ranked)}
    return sorted(set(kit) | set(rest), key=lambda n: (order.get(n, len(ranked)), n))


#: prompt header for the injected block -- names the constraint without claiming completeness.
GROUNDING_HEADER = (
    "Reference: REAL Hexagon intrinsics relevant to this kernel (from the SDK header). "
    "Use these exact names/signatures; do NOT invent intrinsic names.")


def grounding_block(prompt, dtype, index, k=32, mechanisms=None):
    """The full text block to prepend to a task prompt, or ``""`` if nothing was retrieved."""
    names = retrieve_for_task(prompt, dtype, index, k=k, mechanisms=mechanisms)
    body = render_signatures(names, index)
    return "%s\n%s\n\n" % (GROUNDING_HEADER, body) if body else ""


def load_signature_index(path):
    """Load an index previously exported by ``python -m hexkernels.gym.intrinsic_signatures``.

    Generation runs on the GPU box, which need not have the Hexagon SDK staged (same split as
    the valid-name JSON used by the intrinsic mask), so the index ships as a file."""
    import json
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _main(argv=None):
    """Export the signature index to JSON so a GPU box without the SDK can ground prompts."""
    import argparse
    import json
    from hexkernels.core.toolchain import DEFAULT_SDK_ROOT

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdk-root", default=os.environ.get("HEXAGON_SDK_ROOT", DEFAULT_SDK_ROOT))
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    index = signature_index(args.sdk_root)
    if not index:
        raise SystemExit("no intrinsic prototypes found under %s" % args.sdk_root)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print("wrote %s (%d intrinsic signatures from %s)" % (args.out, len(index), args.sdk_root))


if __name__ == "__main__":
    _main()
