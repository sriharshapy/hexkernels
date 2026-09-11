"""Choose the next batch's kernels FROM THE HARVEST, not by taste.

    python -m hexkernels.forge.mine --batches 16-25 --out hexbench/forge2/mined.py

WHY THIS EXISTS
---------------
`CLAUDE.md` and `RESUME.md` have carried the same open gap since batch 3:

    "Selection is not yet harvest-DRIVEN, and that gap is open. All kernels
     resolve, but the ops were chosen by hand ... So the harvest currently proves
     provenance and does not choose kernels."

Batches 4-15 narrowed it -- every op came from `coverage`'s measured `covered`
list rather than from recollection -- but a human still picked the THEME and then
looked for ops to fit it. This module closes it: the op list, the order, the
dtype and the size are all derived, and the only human input is how many batches
to mine.

That matters for the corpus's central claim. The provenance chain a reader is
asked to believe is "op list from the operator registry, sizes from the memory
hierarchy, mechanisms computed from those sizes". Until selection is mechanical,
the first link is an assertion about someone's judgement.

WHAT IS DERIVED, AND FROM WHAT
------------------------------
    which ops     `coverage.scan`'s `covered` set -- ops PROVEN expressible by
                  tracing them and emitting C, minus everything already in the
                  corpus, minus anything whose decomposition contains a primitive
                  that fails `mechanism_eligible` (see `_would_be_destroyed`)
    the order     by the harvest's own row order, which is
                  `torch._C._jit_get_all_schemas()` order -- stable, and nobody's
                  preference
    the dtype     from the op's probe dtype: the widest the op accepts, because a
                  narrow dtype is a separate task rather than a cheaper one
    the size      grown until `plan_for` reports the TARGET TIER for that slot

THE TIER PATTERN IS THE ONE DELIBERATE CHOICE, and it is a spread rather than a
preference: each batch of five targets T0, T1, T1, T2, T3. The corpus-wide EDA
found 62% of the first 55 kernels in T1 with only 3 in T0 and 5 in T3, so mining
uniformly would deepen an imbalance the audit had just flagged.

PRE-FILTERING FOR DESTRUCTION IS NOT OPTIONAL. `provenance.assert_proven` runs as
a build stage and DESTROYS a kernel whose graph contains an ineligible primitive.
Three batches have lost a kernel that way -- 6 (permute/slice/expand/arange), 9
(`torch.outer` -> `view`), 15 (`torch.nan_to_num` -> `aten.scalar_tensor`). Each
cost a reference build before the loss was visible. `_would_be_destroyed` runs the
same check on the traced graph BEFORE a spec is emitted, so mining spends no
simulator time on a kernel that cannot be kept.
"""
import argparse
import json
import math
import os
import pathlib
import sys

import torch

from hexkernels.forge.frontend.trace import trace
from hexkernels.forge import coverage as C
from hexkernels.forge import provenance as _prov
from hexkernels.forge.mechanism import (HMX, HMX_MAX_DTYPE_BYTES, TIERS,
                                       mechanism_eligible, plan_for,
                                       primitive_admissible)

REPO = pathlib.Path(__file__).resolve().parents[1]
COVERAGE_CACHE = REPO / "benchmark" / "coverage_cache.json"
DEFAULT_POOL = REPO / "benchmark" / "pool.json"

#: Target tier per slot within a batch. See the module docstring: the corpus-wide
#: EDA found T1 at 62% with T0 at 3 and T3 at 5 of 55, so an even spread is the
#: correction rather than a taste.
TIER_PATTERN = ("T0", "T1", "T1", "T2", "T3")

#: Row shapes tried in order until one lands in the target tier. Widths are all
#: multiples of 64 elements so an fp16 row is a whole number of HVX vectors and a
#: candidate needs no masked store -- there is no masked vector store in this
#: header (`Q6_Q_vsetq_R` exists, nothing consumes it), so a ragged width forces a
#: scalar tail that would dominate a small kernel.
SHAPE_LADDER = (
    (8, 64), (16, 64), (24, 64), (32, 64), (48, 128), (64, 128),
    (96, 128), (128, 256), (192, 256), (256, 256), (256, 384), (320, 384),
    (384, 512), (512, 512), (512, 768), (768, 768), (768, 1024), (1024, 1024),
    (1280, 1024), (1536, 1024), (1024, 2048), (1536, 2048), (2048, 2048),
)

#: SHAPE_LADDER, restricted to rungs whose first dimension is a multiple of 32.
#:
#: Used ONLY when sizing the fp16 twin (and its multi-tier sweep) for an op that
#: earns `hmx` -- never for the ordinary fp32 search, where a non-tile shape is
#: fine because no tile engine is in play. The HMX tile engine works in 32-wide
#: crouton tiles; `mechanism._contracts` grants `hmx` on the SHAPE of the traced
#: graph (parallel loops over a shared reduction) with no alignment check at
#: all, so `SHAPE_LADDER`'s first four rungs -- (8,64), (16,64), (24,64),
#: (48,128) -- would grant `hmx` to a task no witness on the tile engine can
#: ever implement. For the ops this ladder is used for (`linear`-shaped:
#: self and the second operand drawn at the SAME (M, K) because `_args_for`
#: gives every binary op one shared shape -- see its own docstring -- so the
#: decomposition is (M, K) x (K, M) -> (M, M)), M = shape[0] is EVERY
#: dimension that matters: K = shape[1] (already a multiple of 32 for the
#: whole ladder) and the output's (M, M) inherits M from the same value. So
#: filtering on `shape[0] % 32 == 0` is sufficient to keep M, K and N all
#: 32-aligned, without a second, hand-maintained table that could drift from
#: `SHAPE_LADDER` itself.
#:
#: NOT applied to a SPATIAL op's ladder (`spatial_ladder`, used by
#: `_convolution` and the rest of the conv family): that ladder's tile-relevant
#: axis is the CHANNEL count, not a flat (M, K), and its first rung is
#: (1, 4, 8, 8) -- channel 4, not 32-aligned either. Fixing it needs its own
#: derivation (channels are shared with `_PROBE_CHANNELS` and the conv weight
#: shape in `coverage._weight_shape`), which is a different shape family from
#: the one this ladder is filtered from -- flagged, not forced, per the
#: instruction not to force a fix where the two do not obviously line up.
HMX_SHAPE_LADDER = tuple(s for s in SHAPE_LADDER if s[0] % 32 == 0)


#: (N, C, H, W) shapes for a SPATIAL row, growing the same way `SHAPE_LADDER` does.
#:
#: A pooling or convolution op reads its trailing axes as spatial, so it cannot be
#: sized on the flat ladder at all -- every entry there is rank 2 and torch rejects
#: the call. Channels grow before the spatial extent because that is the axis a
#: kernel vectorises along, and the batch stays 1 so the working set is a clean
#: function of C x E x E.
SPATIAL_LADDER_2D = (
    (1, 4, 8, 8), (1, 8, 16, 16), (1, 16, 16, 16), (1, 16, 32, 32),
    (1, 32, 32, 32), (1, 32, 64, 64), (1, 64, 64, 64), (1, 64, 128, 128),
    (1, 128, 128, 128), (1, 128, 256, 256), (1, 256, 256, 256),
)


def spatial_ladder(rank):
    """The shape ladder for a row with `rank` spatial axes: (N, C) + rank extents."""
    if rank == 2:
        return SPATIAL_LADDER_2D
    out = []
    for n, c, h, _w in SPATIAL_LADDER_2D:
        out.append((n, c) + (h,) * rank)
    return tuple(out)


def load_coverage(path=None):
    """The cached scan. Refuses to guess if it is absent: mining from a stale or
    invented op list would defeat the point of mining at all."""
    cache = pathlib.Path(path) if path else COVERAGE_CACHE
    if not cache.exists():
        raise SystemExit(
            f"no coverage cache at {COVERAGE_CACHE}.\n"
            "Run:  python -m hexkernels.forge.coverage  (and cache it), or see "
            "scratchpad/cache_cov.py -- mining needs a MEASURED covered list, "
            "not a guessed one.")
    return json.loads(cache.read_text(encoding="utf-8"))


def corpus_targets():
    """Every aten target already used by a kernel ANYWHERE in the corpus.

    Read by TRACING the existing specs rather than from a hand-kept list, so a
    batch added later cannot drift out of sync with it.

    `all_batches()`, NOT the hand-written mapping. This iterated `BATCHES` (since
    renamed `HANDWRITTEN_BATCHES`, for exactly this reason), which registers only
    the hand-written 1-15, so the "already in the corpus" filter could not see the
    50 MINED kernels -- and mining batch 26 would have re-selected ops already
    built in batches 16-25, silently, with the duplicates only visible to whoever
    compared two selection files by hand. The dedup existed and was correct; its
    scope simply never grew when the mined batches arrived.
    """
    from hexkernels.forge.kernels import all_batches, batch
    seen = set()
    for b in all_batches():
        for spec in batch(b):
            try:
                g = trace(spec.module, spec.args, spec.name)
            except Exception:                      # noqa: BLE001
                continue
            for n in g.nodes:
                seen.add(n.target)
    return seen


def _harvest_index():
    rows = _prov.load_harvest()
    idx = {}
    for r in rows:
        if r.get("namespace") != "aten":
            continue
        idx.setdefault((r["op"], r.get("overload") or ""), r)
    return idx


def _would_be_destroyed(graph, hidx):
    """The primitives in `graph` that `assert_proven` would reject.

    Same predicate as the build stage, run here so mining never spends a
    reference build on a kernel the provenance gate will delete.
    """
    bad = []
    for n in graph.nodes:
        t = n.target
        if not t.startswith("aten."):
            bad.append(t)
            continue
        op, ov = t[len("aten."):].rsplit(".", 1)
        row = hidx.get((op, "" if ov == "default" else ov))
        # `primitive_admissible`, NOT `mechanism_eligible`: this asks whether the
        # primitive may appear INSIDE the selected op's graph, and the selection
        # question is asked separately about the op being selected (see `mine`).
        # Using the selection predicate here destroyed 66 ops over a `permute`.
        if row is None or not primitive_admissible(row):
            bad.append(t)
    if bad:
        return bad
    # VACUITY, checked here for the same reason admissibility is: `assert_proven`
    # destroys a kernel whose every primitive is plumbing, and this function exists so
    # mining never spends a reference build on a kernel that cannot be kept.
    #
    # It was missed when the primitive/selection split landed, and the cost was
    # immediate and visible: `alias_copy` and `atleast_3d` were selected into batch 36,
    # built, and destroyed -- so the batch came out 3 kernels instead of 5. The
    # non-vacuous filter in `mine` proper does not catch these because it tests whether
    # the SIZE grants `hvx`, which a pure copy of the right size does.
    if all(not mechanism_eligible(hidx.get(_key(n.target), {})) for n in graph.nodes):
        return ["<vacuous: every primitive is plumbing, so the kernel computes "
                "nothing and assert_proven would destroy it>"]
    return bad


def _key(target: str):
    """`aten.sum.dim_IntList` -> `("sum", "dim_IntList")`, the harvest index key."""
    if not target.startswith("aten."):
        return (target, "")
    op, ov = target[len("aten."):].rsplit(".", 1)
    return (op, "" if ov == "default" else ov)


def _depth_counts(targets, hidx):
    """`(n_primitives, n_plumbing)` for a decomposition's node-target multiset.

    Depth is recorded, not tiered on. The design rejected a depth AXIS because
    depth does not determine which mechanism a task needs -- but that argument
    only holds if depth stays MEASURABLE, so the counts are taken here from the
    target list a caller already has (the alias/dedup signature IS that list;
    counting it spends no extra trace) rather than re-derived later.

    `n_plumbing` reuses `_would_be_destroyed`'s own predicate -- a primitive is
    plumbing exactly when `hidx` has no row for it, or the row is not
    `primitive_admissible` -- evaluated here on target STRINGS instead of
    `Node` objects, which is all `targets` (a `graph_signature`/alias-shape
    trace's sorted target tuple) or a real traced `Graph`'s node list ever
    reduces to.
    """
    n_plumbing = 0
    for t in targets:
        row = hidx.get(_key(t))
        if row is None or not primitive_admissible(row):
            n_plumbing += 1
    return len(targets), n_plumbing


def _packet(op, overload):
    pk = getattr(torch.ops.aten, op)
    return getattr(pk, overload or "default")


class _Call(torch.nn.Module):
    """One registry call, so every primitive in the traced graph is attributable
    to this op's decomposition and nothing else.

    A SYNTHESISED NON-TENSOR ARGUMENT IS BAKED IN, NOT PASSED. `trace()` reads
    `.shape` off every example argument, so interleaving the plan's values into the
    argument tuple failed with "'int' object has no attribute 'shape'" for 52 of the
    78 synth-reachable ops -- `_softmax`, `_log_softmax`, `_safe_softmax`, `all.dim`,
    `any.dim`, `cumsum`, `flip`, and every `.Scalar` overload.

    Baking is also the right ANSWER and not merely a working call: only tensors
    become graph placeholders, so the emitted kernel's signature is the tensor list
    and a synthesised `dim` or `alpha` appears as a literal in the C -- which is what
    it is.

    THE SAME FIX WAS NEEDED IN `coverage._Call`, and that is worth saying: three
    modules keep their own one-call wrapper (here, `coverage`, and `mined.MinedCall`)
    because they have different lifetimes, and the argument-baking rule has to hold
    in all three or the prober and the miner disagree about what an op even is.
    """

    def __init__(self, fn, synth_plan=None):
        super().__init__()
        self.fn = fn
        self.synth_plan = synth_plan

    def forward(self, *args):
        if self.synth_plan is None:
            return self.fn(*args)
        it = iter(args)
        return self.fn(*(next(it) if kind == "t" else v
                         for kind, v in self.synth_plan))


def _args_for(sig, shape, dtype, plan=None, op=None, names=None):
    """Example arguments for a probe or a size search.

    Returns a LIST of candidate argument tuples to try, in schema order --
    almost always exactly one, except for a binary self/mat2-shaped row
    (`mm`, `bmm`; see below), where it returns two and the caller (already
    tracing and catching exceptions -- `size_for_tier`, `graph_signature`)
    keeps whichever one torch accepts.

    `plan` is `coverage.synth_signature`'s per-position argument plan, present for a
    row whose required arguments are NOT 1-2 plain tensors -- `scaled_dot_product_
    attention` needs three tensors then a mask and a dropout probability, and
    `rnn_relu_cell` needs four tensors and two biases. The tensors are drawn exactly
    as they always were and the non-tensor values are taken from the plan, so a
    synth-reached op differs from an ordinary one only in the ARGUMENT LIST and not
    in how its data is generated.

    `names` is the row's REQUIRED TENSOR argument names in schema order, consulted
    ONLY when there is no plan and `sig == "binary"` -- exactly `mm`/`bmm`, the two
    ops `coverage.probe_signature` already accepts without inventing a value. Before
    this, EVERY binary op (matmul included) drew both operands at `[shape] * 2`, so
    `mm((8,64),(8,64))` -- not a valid matmul, since `self`'s last dim must equal
    `mat2`'s first -- was the only shape ever tried, and `mm`/`bmm`/`addmm`/`baddbmm`
    could never be MINED into a task even once Route B made them PROBEABLE
    (`coverage.coverage_cache.json`'s `covered`/`covered_synth`). This was the
    IDENTICAL defect coverage.probe's own binary path had before Route B fixed it
    there -- so the fix here calls `coverage.contraction_tshapes` (PUBLIC, and
    parameterised on `(m, k)` for exactly this reason) rather than re-deriving the
    (M, K) x (K, N) geometry a second time in a second module, which is how the two
    would eventually disagree without anyone noticing.
    """
    # PER-POSITION SHAPES, like `coverage.probe`. A plan entry may carry a shape for
    # its tensor -- a convolution weight is (Cout, Cin, KH, KW) and is NOT the shape of
    # the input. Threading it through `coverage` and not through here is the SAME bug
    # this pipeline has now produced four times: two places that each construct the
    # call, fixed in one. It presented as `slow_conv_transpose2d` sizing to no tier at
    # all, because the weight was drawn at the input's shape and torch rejected it --
    # a failure that reads as "this op cannot be sized" and is really "we called it
    # wrong".
    if plan is not None:
        n = sum(1 for kind, _v in plan if kind == "t")
        tshape_candidates = [[(sh or shape) for kind, sh in plan if kind == "t"]]
    else:
        n = 1 if sig == "unary" else 2
        tshape_candidates = None
        if n == 2 and names and len(names) == 2:
            tshape_candidates = C.contraction_tshapes(
                names, shape[0], shape[1] if len(shape) > 1 else shape[0])
        if not tshape_candidates:
            tshape_candidates = [[shape] * n]

    def _build(tshapes):
        if dtype is torch.bool:
            return tuple(torch.ones(s_, dtype=dtype) for s_ in tshapes)
        if dtype.is_floating_point:
            # away from 0 and 1: log, rsqrt and div are undefined or degenerate there,
            # and a NaN golden fails for a reason unrelated to expressibility.
            #
            # AND THROUGH THE SAME DOMAIN SHIFT THE SPEC WILL USE. `mined._args` applies
            # `_domain_shift` (acos to [-0.9, 0.9], acosh to [1.5, 2.5), logit to
            # [0.05, 0.95)); this drew the raw range, so the miner was evaluating a
            # DIFFERENT task from the one it was about to emit -- and would have rejected
            # `logit` for a NaN the spec never produces. Same class of mistake as
            # comparing two graph signatures computed at different dtypes.
            from hexkernels.forge.mined import _domain_shift
            return tuple(_domain_shift(op, torch.rand(s_, dtype=dtype) + 0.5)
                        for s_ in tshapes)
        return tuple(torch.randint(1, 5, s_, dtype=dtype) for s_ in tshapes)

    # ONLY THE TENSORS. The plan's values are baked into `_Call`, so they are
    # constants in the traced graph rather than placeholders -- see `_Call`.
    return [_build(tshapes) for tshapes in tshape_candidates]


def _dtype_bytes(dtype):
    return {torch.float32: 4, torch.float16: 2, torch.int32: 4,
            torch.int8: 1, torch.uint8: 1, torch.bool: 1,
            torch.int64: 8}.get(dtype, 4)


def _golden_is_computable(mod, args) -> bool:
    """Can this module actually be RUN on these arguments, finitely?

    Tracing proves the graph exists; it does not prove torch will execute the op at
    this dtype, nor that the result is finite. A non-finite golden is worse than a
    wrong one: `nan != nan`, so every element fails and the failure looks like a
    kernel bug rather than an out-of-domain input.
    """
    try:
        with torch.no_grad():
            out = mod(*args)
    except Exception:                              # noqa: BLE001 - registry is wide
        return False
    outs = out if isinstance(out, (tuple, list)) else (out,)
    for o in outs:
        if not isinstance(o, torch.Tensor):
            continue
        if o.dtype.is_floating_point and not bool(torch.isfinite(o).all()):
            return False
    return True


def size_for_tier(op, overload, sig, dtype, target, hidx, aplan=None, ladder=None):
    """Grow the shape until `plan_for` reports `target`.

    Returns `(shape, mechanism_plan, argument_plan)` or None. THE ARGUMENT PLAN IS
    RETURNED because this function RE-DERIVES it per ladder rung for a spatial row --
    the weight's channels and kernel extent come from the shape being tried -- and the
    caller was storing the plan it passed IN, computed once at the probe's defaults.
    That put a weight shape in `mined_selection.json` that contradicted the input shape
    recorded beside it. Sixth instance of one bug class this session; see GOAL_500.md.

    The size is DERIVED, not chosen: the tier is a comparison of the working set
    against this target's L1D/L2/VTCM, so asking for a tier and solving for the
    shape is the same statement as asking for a shape and reporting the tier --
    run in the direction that makes the corpus's spread controllable.

    `ladder`, if given, OVERRIDES the flat (non-spatial) ladder -- used by the
    fp16-twin search to require `HMX_SHAPE_LADDER` (32-aligned) instead of the
    ordinary `SHAPE_LADDER`, since a shape granted `hmx` has to be one the tile
    engine can actually consume. Ignored for a SPATIAL row, which always uses
    `spatial_ladder(srank)` regardless -- see `HMX_SHAPE_LADDER`'s own comment
    for why that ladder is not (yet) filtered the same way.
    """
    fn = _packet(op, overload)
    # A SPATIAL ROW NEEDS A RANK-AWARE LADDER **AND** ARGUMENT VALUES RE-SYNTHESISED
    # PER SHAPE. Both, not either: the flat ladder is rank 2 so torch refuses the call
    # outright, and reusing the prober's values (derived at extent 8) against a
    # 256-wide shape would give `upsample` an output_size of 4 -- a 64x downsample,
    # a legal call and a task nobody meant to build. This is the same
    # coherence requirement that capped `MAX_SYNTH_ARGS` at 5.
    srank = None
    row = hidx.get((op, overload))
    if aplan is not None and row is not None:
        srank = C.spatial_rank(row)
    # BINARY CONTRACTION OPERANDS: a plain (no synth plan) binary row whose
    # required tensor arguments are named `self`/`mat2` (`mm`, `bmm`) needs
    # (M, K) x (K, N), not one shared shape for both operands -- see
    # `_args_for`'s own docstring. The argument NAMES do not change across
    # ladder rungs, so this is resolved once per op rather than inside the
    # loop below.
    names = None
    if aplan is None and sig == "binary" and row is not None:
        names = [a["name"] for a in C.required_args(row) if a.get("is_tensor")]
    ladder = spatial_ladder(srank) if srank else (ladder or SHAPE_LADDER)
    for shape in ladder:
        if srank:
            # re-derive the plan at THIS shape's extent
            # extent AND channels from THIS shape, so the weight's Cin matches the
            # input's channel axis at every rung of the ladder
            aplan = C.synth_signature(row, extent=shape[-1],
                                      channels=shape[1] if len(shape) > 2 else None)
            if aplan is None:
                return None
        mod = _Call(fn, aplan)
        # `aplan`, not `plan`: the ARGUMENT plan. The mechanism `plan_for` result
        # below is also called `plan`, and when this parameter shared that name it
        # was silently overwritten after the first shape on the ladder -- so the
        # second iteration passed a mechanism Plan where an argument plan belonged.
        # A shadowed parameter in a loop is invisible on the first pass, which is
        # exactly when a smoke test looks fine.
        #
        # WRAPPED, and this is not optional. `_args_for` ALLOCATES the example
        # tensors -- unlike everything else in this loop, which only traces or
        # inspects a graph. Found live: the MULTI-TIER SWEEP (mining an
        # already-hmx-qualified op at tiers other than the one its fp32 parent
        # landed on) reached a ladder rung whose synthesised shape asked for an
        # 8 GB allocation and raised `RuntimeError` from the allocator, which
        # was uncaught here and took the whole batch driver down mid-run --
        # not a `trace()`-time failure (that branch was already guarded) but
        # one step earlier, building the arguments `trace()` would have been
        # given. A single ladder rung being too large for this machine is a
        # fact about that rung, not a reason to lose every already-computed
        # pick in the run.
        try:
            arg_candidates = _args_for(sig, shape, dtype, aplan, op, names=names)
        except Exception:                          # noqa: BLE001
            return None
        # TRY EACH CANDIDATE IN ORDER, exactly like `coverage.probe` does for the
        # same self/mat2 ambiguity: `mm` (rank 2) and `bmm` (rank 3) declare
        # identical parameter names, so the ambiguity can only be resolved by
        # which shape torch actually accepts, never by matching the op's name.
        g, args = None, None
        for cand_args in arg_candidates:
            try:
                g = trace(mod, cand_args, f"mine_{op}")
            except Exception:                      # noqa: BLE001
                g = None
                continue
            args = cand_args
            break
        if g is None:
            return None
        bad = _would_be_destroyed(g, hidx)
        if bad:
            return None                            # would be destroyed; skip the op
        # A GRAPH THAT TRACES IS NOT A TASK THAT HAS A GOLDEN, and this checked only
        # the former. `logaddexp` traces at int32 and then torch refuses to execute it
        # ("not implemented for 'Int'"), so it was selected, built, and failed its own
        # reference on a dtype the op does not support. `logit` traces at any dtype and
        # returns NaN for more than half a [0.5, 1.5) draw, which fails every element
        # because `nan != nan`.
        #
        # Both are properties of the (op, dtype, draw) triple, so they belong in the
        # search that picks it -- one small forward pass here moves a whole class of
        # failure from "the reference build fails" to "that dtype is never selected".
        # The domain table in `mined._DOMAIN` is still the right fix for an op worth
        # KEEPING at a shifted range; this is the backstop for the rest.
        if not _golden_is_computable(mod, args):
            continue                               # try the next shape/dtype
        try:
            plan = plan_for(g, dtype_bytes=_dtype_bytes(dtype))
        except Exception:                          # noqa: BLE001
            return None
        # A dtype the emitted C cannot express is not a cheaper task, it is a
        # broken one. Checked on the FIRST shape only: the operators in the body
        # do not change with the size, and compiling a 2 M-element reference to
        # learn about a bitwise operator would be absurd.
        if shape == ladder[0]:
            from hexkernels.forge.frontend.emit import emit_c
            try:
                if _needs_integer(emit_c(g)) and dtype.is_floating_point:
                    return None
            except Exception:                      # noqa: BLE001
                return None
        if plan.tier == target:
            return shape, plan, aplan
        if plan.tier > target:                     # overshot; this op cannot hit it
            return None
    return None


PROBE_DTYPE_ORDER = (torch.float32, torch.float16, torch.int32, torch.int8)


def hmx_twin_dtype(graph, tgt=None):
    """The narrow dtype that would make this graph an HMX task, or None.

    `PROBE_DTYPE_ORDER` tries fp32 first and breaks on the first dtype that
    sizes, so every float op lands at 4 bytes and `mechanism.plan_for` refuses
    HMX -- correctly, since there is no fp32 tile multiply. That starved the
    HMX column to zero across 925 selected specs.

    This does not reorder the probe. It asks a narrower question: holding the
    graph fixed, does dropping to 2 bytes ADD `hmx`? Only a contraction can,
    because `mechanism._contracts` is what gates it. So the answer doubles as
    contraction detection, and the fp32 form stays in the set as the negative
    control that proves the entitlement rule refuses as well as grants.
    """
    if HMX in plan_for(graph, dtype_bytes=4, tgt=tgt).mechanisms:
        return None  # already granted at 4 bytes: nothing to twin
    if HMX in plan_for(graph, dtype_bytes=HMX_MAX_DTYPE_BYTES,
                       tgt=tgt).mechanisms:
        return torch.float16
    return None

#: Integer-only C operators. An emitted body containing one of these cannot be
#: compiled against a float buffer.
#:
#: THIS EXISTS BECAUSE `covered` DOES NOT MEAN `COMPILES`. `coverage.probe` checks
#: that C can be EMITTED and the nest ANNOTATED -- it never invokes the compiler.
#: So `__and__`, `__or__`, `__xor__` and `bitwise_not` were all reported covered at
#: float32, and their emitted references contain `(a & b)` on `float` operands,
#: which is not valid C. Mining them at float32 would have spent four reference
#: builds discovering that.
#:
#: The cheap, general check is to COMPILE the emitted reference -- see
#: `_reference_compiles`. This table is the fast pre-filter that avoids even
#: trying a dtype that cannot work.
INT_ONLY_OPS = ("&", "|", "^", "~", "<<", ">>", "%")


def _needs_integer(kernel_c: str) -> bool:
    """True if the emitted body uses an operator only defined on integers.

    Looks at the STATEMENT bodies, not the whole file: index arithmetic uses `*`
    and `+` freely and array subscripts are irrelevant, but a bitwise operator in a
    value expression pins the dtype.
    """
    for line in kernel_c.splitlines():
        st = line.strip()
        if not st.startswith("v_") or "=" not in st:
            continue
        rhs = st.split("=", 1)[1]
        # drop subscripts, which contain only index arithmetic
        depth, cleaned = 0, []
        for ch in rhs:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            elif depth == 0:
                cleaned.append(ch)
        expr = "".join(cleaned)
        if any(op in expr for op in INT_ONLY_OPS):
            return True
    return False

#: Shape used only for the alias signature -- small and fixed, because the
#: signature must depend on the OP and not on the size.
ALIAS_SHAPE = (8, 64)


def graph_signature(op, overload, sig, hidx, dtype=torch.float32, plan=None):
    """The multiset of primitives an op decomposes to, as a hashable key.

    `dtype` IS PART OF THE SIGNATURE, and defaulting it was a bug. This traced
    every candidate at float32 while `corpus_signatures` traces each existing spec
    at its REAL dtype, so the two sides of the comparison were computed under
    different conditions -- and `isclose`, which the integer-operator gate builds at
    int32, was admitted a SECOND time in batch 26 because its int32 decomposition
    (which carries an int->float `_to_copy`) never matched the float32 one the
    filter had computed for it. Structurally the same defect as comparing raw map
    text against normalised map text: a check that cannot match is not a check.
    Callers must pass the dtype the kernel will actually be built at, which means
    the alias test has to run AFTER the size/dtype search rather than before it.

    THIS IS THE ALIAS FILTER, and it is needed because the registry is full of
    spellings of the same operation. Mining the `covered` list in harvest order
    produced `absolute` alongside `abs`, `arccos` alongside `acos`, and
    `arcsin`/`arcsinh`/`arctan`/`arctanh`/`divide`/`fix` alongside their canonical
    names -- twelve of the first fifty picks were duplicates of another pick or of
    something already in the corpus.

    A name-based blocklist would be a judgement call and would go stale. Two ops
    that DECOMPOSE TO THE SAME PRIMITIVE GRAPH are the same task for this
    pipeline's purposes, whatever they are called, and that is checkable: trace
    both and compare. It also correctly collapses ops that are not documented
    aliases but happen to lower identically, which is the same argument.
    """
    # RANK-AWARE, LIKE `size_for_tier`. `ALIAS_SHAPE` is rank 2, so tracing a SPATIAL
    # op at it throws and this returned None -- which the caller reads as "no
    # signature" and skips the op. That is how eight expressible upsample kernels were
    # silently unselectable AFTER the emitter, the prober and the size search all
    # worked: the third module that keeps its own call-construction had not been
    # taught the same thing. Exactly the failure mode the synth-scalar fix had:
    # `coverage`, `mine` and `mined` each build a call, and fixing two of three leaves
    # them disagreeing about what the op is.
    row = hidx.get((op, overload))
    srank = C.spatial_rank(row) if (plan is not None and row is not None) else None
    shape = ((1, 4) + (8,) * srank) if srank else ALIAS_SHAPE
    if srank:
        plan = C.synth_signature(row, extent=shape[-1],
                                 channels=shape[1] if len(shape) > 2 else None) or plan
    # SAME BINARY-CONTRACTION AMBIGUITY AS `size_for_tier`: `bmm`'s alias
    # signature cannot be traced at `ALIAS_SHAPE`'s rank-2 candidate, so
    # without this it always returned None here (a TRACE failure, read by
    # the caller as "no signature") even after `size_for_tier` had already
    # sized the op successfully.
    names = None
    if plan is None and sig == "binary" and row is not None:
        names = [a["name"] for a in C.required_args(row) if a.get("is_tensor")]
    try:
        mod = _Call(_packet(op, overload), plan)
        arg_candidates = _args_for(sig, shape, dtype, plan, op, names=names)
    except Exception:                              # noqa: BLE001
        return None
    g = None
    for args in arg_candidates:
        try:
            g = trace(mod, args, f"sig_{op}")
            break
        except Exception:                          # noqa: BLE001
            g = None
            continue
    if g is None:
        return None
    return tuple(sorted(n.target for n in g.nodes))


def corpus_signatures():
    """Graph signatures of every kernel already in the corpus, so a mined op that
    merely re-spells an existing task is skipped.

    `all_batches()` for the same reason as `corpus_targets`: scoped to the
    hand-written mapping
    this could not see the mined half, so the alias filter -- the thing that makes
    "50 kernels" mean "50 tasks" -- would have re-admitted `abs` beside `absolute`
    across batch series instead of only within one.
    """
    from hexkernels.forge.kernels import all_batches, batch
    out = set()
    for b in all_batches():
        for spec in batch(b):
            try:
                g = trace(spec.module, spec.args, spec.name)
            except Exception:                      # noqa: BLE001
                continue
            out.add(tuple(sorted(n.target for n in g.nodes)))
    return out


def mine(n_batches, start_batch, per_batch=5, exclude=None, synth=False,
         coverage_path=None):
    """Select `n_batches * per_batch` kernels. Returns a list of batch lists.

    `synth=True` also walks `covered_synth` -- rows reachable only once the prober
    invents values for arguments that are not plain tensors. Kept behind a flag and
    walked AFTER the plain list, so the mechanical order is preserved and a synth
    row can never displace one that needed no invention.

    THAT DISTINCTION IS THE PROVENANCE ARGUMENT, not bookkeeping. A `covered` row's
    task is fully determined by the registry: the schema says what the arguments are
    and the harvest says the op exists. A `covered_synth` row's task additionally
    depends on values THIS PIPELINE chose -- `scaled_dot_product_attention` needs a
    dropout probability and an is_causal flag that no schema fixes. The op's origin
    is unchanged and still PyTorch's, which is what provenance asserts; but the
    kernel is one instance of a family, so every such entry records the plan it was
    built from and `PROVENANCE.md` can say which values were invented and which were
    read. A reader can check the difference only if it is written down.
    """
    cov = load_coverage(coverage_path)
    hidx = _harvest_index()
    used = set(exclude or ())
    used |= corpus_targets()

    # WHY A REJECTION TRACE LIVES IN THE LOOP ITSELF.
    #
    # Twice now a row that passed every filter when checked STANDALONE was still not
    # picked, and both times the cause was a condition that only exists in this
    # function: first `graph_signature` returning None at a rank-2 `ALIAS_SHAPE`, then
    # the `max_pool` family. Re-deriving these conditions in a throwaway script
    # reproduces the checks and NOT the loop -- so it agrees with the caller and both
    # are wrong together. `MINE_DEBUG=1` prints the point of rejection from inside the
    # real control flow, which is the only place that cannot disagree with itself.
    dbg = os.environ.get("MINE_DEBUG")

    def _rej(key, why):
        if dbg:
            print(f"  REJECT {key:44s} {why}", file=sys.stderr)

    picked, batches = [], []
    seen_sigs = corpus_signatures()
    slot = 0
    keys = list(cov["covered"])
    if synth:
        keys += [k for k in cov.get("covered_synth", ()) if k not in set(keys)]
    # A SECOND PASS WAS TRIED TWICE AND MEASURED AT ZERO BOTH TIMES.
    #
    # Each key is tried against the CURRENT slot's tier as encountered, so an op can be
    # discarded for arriving while the slot wants a tier it cannot reach. Walking
    # `keys + keys` fixes that in principle. Measured twice -- once when the T3 slot had
    # no candidate at all, and again after `digamma`, `i0` and `erfinv` were added on
    # the theory that elementwise ops could be grown into the empty T2/T3 slots. Both
    # runs covered the full doubled list and picked exactly the same ops as one pass.
    #
    # The second result explains the first. The elementwise ops that CAN reach any tier
    # sort early, get offered T0/T1, and are taken there -- so by the time the empty
    # slots come up they are already used, and what remains genuinely cannot reach T2 or
    # T3 at any shape on the ladder. A second pass re-offers the same unsuitable
    # remainder. It doubles mining time for nothing, so it does not ship.
    #
    # What WOULD change this is displacement-aware selection (letting a later, larger
    # op take a tier from an earlier, smaller one that can sit elsewhere), which is a
    # real change to what "mechanically selected" means and is not worth making to
    # save one op.
    for key in keys:
        # ENTRY TRACE, unconditional. A rejection-only trace cannot distinguish
        # "never visited" from "rejected untraced" from "picked", and inferring which
        # from the counts produced two wrong diagnoses in a row. A line per key on
        # ENTRY settles it by observation.
        if dbg:
            print(f"  ENTER  {key}", file=sys.stderr)
        if len(picked) >= n_batches * per_batch:
            if dbg:
                print(f"  BREAK  quota reached at {key}", file=sys.stderr)
            break
        op, overload = key.rsplit(".", 1)
        overload = "" if overload == "default" else overload
        target_name = f"aten.{op}.{overload or 'default'}"
        if target_name in used:
            _rej(key, "already in the corpus")
            continue
        row = hidx.get((op, overload))
        if row is None or not mechanism_eligible(row):
            _rej(key, "not mechanism_eligible")
            continue
        sig = C.probe_signature(row)
        aplan = None
        if sig is None:
            if not synth:
                continue
            # A synth row: the per-position argument plan IS part of the task now,
            # so it is carried through the size search, the alias signature and the
            # committed selection rather than re-derived at each step.
            try:
                aplan = C.synth_signature(row)
            except Exception:                      # noqa: BLE001
                aplan = None
            if aplan is None:
                _rej(key, "no synth plan")
                continue
        # EARLIEST-FIT ACROSS ALL UNFILLED SLOTS WAS TRIED AND MEASURED AT ZERO.
        #
        # The tier pattern is the SPREAD the corpus should have, and trying only the
        # current slot quietly made it a per-op requirement -- an op reaching T3 but not
        # T2 was dropped while T3 sat empty. Implemented (offer each key every unfilled
        # slot, earliest first) and run over the full key list: the same three picks.
        # So the ~342 unpicked keys were each offered T2 AND T3 and none fit.
        #
        # That settles it: batch 49 stalls because NO REMAINING EXPRESSIBLE OP CAN BE
        # SIZED TO T2 OR T3, which is a property of the ops, not of the rule. A
        # selection-semantics change with zero measured benefit does not ship -- and
        # selection is the corpus's central claim, so it is the last thing to change on
        # a hunch.
        # THE TIER PATTERN IS A PREFERENCE, NOT A REQUIREMENT (user decision).
        #
        # It used to be a hard filter: an op that could not reach the slot's tier was
        # DISCARDED, even with other slots empty and nothing else able to fill them.
        # Measured cost -- batch 50 sat at 4 of 5 through FOUR consecutive correct
        # emitters, because everything new was a small elementwise function that could
        # only reach T0/T1, so each arrival displaced an earlier pick instead of adding
        # one. Two attempts to fix it within the old rule (a second pass over the key
        # list; earliest-fit across the unfilled slots) both measured ZERO, because the
        # binding constraint was the TIER, not the slot ordering.
        #
        # Now: try the slot's intended tier FIRST, so the spread still happens wherever
        # it can, then fall back to any other tier the op can reach. The pattern still
        # shapes the corpus; it no longer throws work away.
        #
        # WHAT THIS COSTS, stated because a deliberate choice is being relaxed: the
        # corpus-wide EDA that motivated the spread found 62% of the first 55 kernels
        # in T1, with 3 in T0 and 5 in T3. Falling back will skew that way again, since
        # small elementwise ops are what remains unbuilt. Each entry records the tier it
        # ACTUALLY got, so the real distribution stays measurable rather than being
        # inferred from the pattern.
        preferred = TIER_PATTERN[slot % per_batch]
        got = target = None
        for cand in [preferred] + [t for t in ("T0", "T1", "T2", "T3")
                                   if t != preferred]:
            for dtype in PROBE_DTYPE_ORDER:
                got = size_for_tier(op, overload, sig, dtype, cand, hidx, aplan)
                if got:
                    break
            if got:
                target = cand
                break
        if not got:
            _rej(key, "cannot be sized to ANY tier")
            continue
        shape, plan, used_aplan = got
        # NON-VACUOUS FILTER: an op granted no COMPUTE mechanism is pure data
        # movement. Mining produced `atleast_1d`, `atleast_2d`, `fliplr` and
        # `conj_physical`, all of which planned to `l2fetch` alone -- at rank 2
        # `atleast_2d` is the identity. CLAUDE.md's own words: the metadata ops
        # "emit perfectly and teach nothing". A task with nothing to accelerate
        # cannot report whether an accelerator was used.
        if "hvx" not in plan.mechanisms:
            _rej(key, "vacuous: no compute mechanism")
            continue
        # ALIAS FILTER: same decomposition means same task, whatever the spelling.
        # AFTER the dtype search, not before: the decomposition depends on the
        # dtype (an integer operand inserts a `_to_copy` a float one does not), so
        # a signature computed at a guessed dtype cannot be compared against one
        # computed at the real dtype. See `graph_signature`.
        gsig = graph_signature(op, overload, sig, hidx, dtype, used_aplan)
        # TWO CAUSES, TWO LABELS. "no graph signature, or alias-duplicate" conflated
        # a TRACE FAILURE with a genuine duplicate, and those call for opposite
        # responses: the first is a bug in our call construction, the second means the
        # op is already in the corpus under another name. Reporting them together is
        # how `graph_signature`'s rank-2 `ALIAS_SHAPE` blocker stayed invisible.
        if gsig is None:
            _rej(key, "graph_signature returned None (the TRACE failed)")
            continue
        if gsig in seen_sigs:
            _rej(key, "alias-duplicate of something already selected or built")
            continue
        seen_sigs.add(gsig)
        if dbg:
            print(f"  PICK   {key}", file=sys.stderr)
        # Depth is recorded, not tiered on. The design rejected a depth AXIS
        # because depth does not determine which mechanism a task needs -- but
        # that argument only holds if depth stays MEASURABLE, so the count is
        # taken here from `gsig`, the alias signature already traced above,
        # rather than re-traced later. Intensity is recorded, not yet entitled
        # on: the measured crossover is ~3 vector ops/element, so recording it
        # now means the set can be re-labelled once the silicon measurement
        # lands, without re-selecting.
        _n_prim, _n_plumb = _depth_counts(gsig, hidx)
        picked.append({
            "op": op, "overload": overload, "sig": sig,
            # None for an ordinary row; the argument plan for a synth one, so the
            # selection records WHICH values were invented rather than leaving that
            # to be rediscovered.
            "synth_plan": used_aplan,
            "dtype": str(dtype).replace("torch.", ""),
            "dtype_bytes": _dtype_bytes(dtype),
            "shape": list(shape), "tier": plan.tier,
            "mechanisms": sorted(plan.mechanisms),
            "working_set": plan.working_set_bytes,
            "schema": row.get("raw", "")[:200],
            "accessor": row.get("accessor", ""),
            "n_primitives": _n_prim,
            "n_plumbing": _n_plumb,
            "ops_per_element": (_n_prim / max(1, math.prod(shape))
                                if shape else None),
        })
        # HMX TWIN. mine.py's own docstring: "a narrow dtype is a SEPARATE task
        # rather than a cheaper one". PROBE_DTYPE_ORDER tries fp32 first and
        # breaks, so every float op landed at 4 bytes and mechanism.plan_for
        # refused hmx -- correctly, there is no fp32 tile matmul. That starved
        # the hmx column to 0 across 925 specs.
        #
        # Gated on whether a plan solved at fp16 GRANTS hmx, never on the op
        # name: name matching would miss contractions and admit non-contractions,
        # and mechanical selection is the corpus's whole claim. The fp32 entry
        # above STAYS -- it is the negative control that proves the entitlement
        # rule refuses as well as grants.
        #
        # DEDUP KEY MUST BE DTYPE- AND TIER-QUALIFIED, and this is not the
        # same int-vs-float case the ALIAS FILTER comment above already
        # covers. `graph_signature` is both dtype-BLIND and SHAPE-blind (it
        # keys on node TARGETS only): a float contraction decomposes to the
        # same targets at fp32 and fp16, and at T0 and at T3 -- unlike an
        # integer operand, which inserts an extra `_to_copy` node. So a bare
        # `_tsig`, or one qualified by dtype alone, is byte-for-byte the same
        # value as the parent's `gsig` (dtype-blind) or as another tier's twin
        # of the same op (shape-blind), and checking either weaker key here
        # made a twin collide with its own parent (fix round 1, `d8ff55e`) or
        # would make a T3 twin collide with the T0 twin already added for the
        # same op (this round). A twin is a DIFFERENT TASK from its fp32
        # parent, and a T3 twin is a DIFFERENT TASK from a T0 twin of the same
        # op -- `task_id` is already `(op, overload, dtype, tier)`, and the
        # SAME contraction measured L1-resident versus streamed from DRAM
        # with staging is precisely what the size axis exists to measure, not
        # an alias of one task found at one size. So every twin is keyed on
        # `(signature, dtype, tier)`. Two fp16 twins of the same decomposition
        # AND the same tier still collapse into one (same qualified key); a
        # twin never collapses into its fp32 parent, nor into a twin of the
        # same op at a different tier (different key each time).
        #
        # THE PARENT'S KEY STAYS BARE, unchanged from `d8ff55e`: `seen_sigs`
        # is seeded from `corpus_signatures()`, which returns bare tuples for
        # the existing corpus, and qualifying the parent's key here would
        # stop it matching those -- weakening the alias filter this function
        # already relies on for the single-dtype path.
        #
        # MULTI-TIER SWEEP: tried at the CURRENT slot's target tier first
        # (preserving where the fp32 parent already proved this op sizes),
        # then at the three tiers not yet tried. `size_for_tier` is called
        # again only for an op that JUST proved (at the tier just tried) that
        # its fp16 form earns `hmx` -- not for the ~350 ops that never will,
        # so this stays cheap relative to the fp32 search that dominates the
        # rest of the loop.
        if dtype is not torch.float16:
            for _tier in (target,) + tuple(t for t in TIERS if t != target):
                _twin = size_for_tier(op, overload, sig, torch.float16,
                                      _tier, hidx, aplan,
                                      ladder=HMX_SHAPE_LADDER)
                if not _twin:
                    continue
                _tshape, _tplan, _taplan = _twin
                if HMX not in _tplan.mechanisms or "hvx" not in _tplan.mechanisms:
                    # The FIRST tier tried (the parent's own) not granting
                    # `hmx` means this op is not a contraction at all --
                    # trying the other three tiers would only spend ladder
                    # searches on ops that were never going to qualify, so
                    # stop rather than sweep them.
                    if _tier == target:
                        break
                    continue
                _tsig = graph_signature(op, overload, sig, hidx,
                                        torch.float16, _taplan)
                _tkey = (_tsig, "float16", _tplan.tier) if _tsig else None
                if _tkey and _tkey not in seen_sigs:
                    seen_sigs.add(_tkey)
                    _tn_prim, _tn_plumb = _depth_counts(_tsig, hidx)
                    picked.append({
                        "op": op, "overload": overload, "sig": sig,
                        "synth_plan": _taplan,
                        "dtype": "float16", "dtype_bytes": 2,
                        "shape": list(_tshape), "tier": _tplan.tier,
                        "mechanisms": sorted(_tplan.mechanisms),
                        "working_set": _tplan.working_set_bytes,
                        "schema": row.get("raw", "")[:200],
                        "accessor": row.get("accessor", ""),
                        "n_primitives": _tn_prim,
                        "n_plumbing": _tn_plumb,
                        "ops_per_element": (_tn_prim / max(1, math.prod(_tshape))
                                           if _tshape else None),
                    })
        used.add(target_name)
        slot += 1

    # THE REMAINDER IS KEPT, NOT SWALLOWED. This used to return only WHOLE
    # per_batch-sized groups and stash the leftover on `mine.remainder`, which
    # `main()` never read or persisted -- so up to `per_batch - 1` valid,
    # already-selected picks were silently discarded every time the covered
    # list ran out (or the quota was reached) mid-batch. That is a real defect
    # regardless of which ops it hit: the grouping step here was silently
    # discarding picks that selection had already made, which is the one
    # thing a "grouping" step must never do. A caller told "mined 0 batches,
    # the covered list ran out of ops" when FOUR ops were selected and
    # discarded has been told something false -- see the git history of this
    # exact comment for the three debugging rounds that cost.
    #
    # `gelu` is a CANDIDATE VICTIM, not a proven one: it passes every
    # selection filter (mechanism-eligible, sizes to a tier, no signature
    # collision) and was still missing from the corpus, which is consistent
    # with landing in a trailing partial group that got dropped here -- but
    # that causal link was never tested (no run was captured mid-mine to show
    # `gelu` actually sitting in the discarded remainder), so it is a theory,
    # not a finding. `gelu` stays absent from the pool and in `KNOWN_ABSENT`;
    # this fix stands on the defect being real, not on that theory being
    # confirmed. The fix is to keep every chunk, full or not: selection order
    # and dedup (both above, untouched) decide WHICH picks exist; this loop
    # only decides how they are grouped, and grouping must never be the
    # reason a pick disappears.
    for i in range(0, len(picked), per_batch):
        batches.append(picked[i:i + per_batch])
    mine.remainder = []
    return batches


#: The second stage of a fusion must be an EPILOGUE -- the composition real networks
#: and real fusion passes actually produce: a compute op followed by an activation, a
#: normalisation, or a reduction.
#:
#: WHY RESTRICT AT ALL, when ~20,000 arbitrary pairs are available and 94% of them
#: trace: because a corpus is judged by what it teaches, not by its row count.
#: `abs then acosh` is a legal graph and a pointless kernel -- it adds no schedule
#: anyone needs and would inflate the dataset while diluting exactly the claim that
#: makes it worth having. `linear then relu` and `conv then softmax` are kernels people
#: actually write and hand-optimise.
FUSION_EPILOGUES = frozenset({
    # activations
    "relu", "gelu", "silu", "mish", "elu", "celu", "selu", "sigmoid", "tanh",
    "hardswish", "hardsigmoid", "hardtanh", "softplus", "leaky_relu", "logsigmoid",
    "log_sigmoid", "hardshrink", "softshrink", "threshold", "clamp", "abs",
    "special_expit", "erf",
    # normalisation / distribution
    "softmax", "_softmax", "log_softmax", "_log_softmax", "_safe_softmax",
    "special_softmax", "special_log_softmax",
    # reductions -- the other classic epilogue
    "sum", "mean", "amax", "amin", "prod", "logsumexp", "var", "std", "norm",
    "linalg_vector_norm", "all", "any", "count_nonzero",
})

#: Ops that return INDICES rather than values. Excluded from the epilogue set even
#: though several are reductions: `arctanh -> argsort` traced, emitted, scheduled and
#: then FAILED its golden, because the fusion's output is int64 and the harness
#: compares it as the float task it was sized as. The type check looks at the
#: intermediate dtype; this is the same question asked of the FINAL output.
FUSION_INDEX_OUTPUTS = frozenset({
    "argsort", "sort", "topk", "argmax", "argmin", "kthvalue", "mode", "median",
    "nanmedian", "max", "min", "unique", "_unique2", "nonzero", "searchsorted",
    "bucketize", "argwhere",
})


_FULL_REDUCTIONS = frozenset({
    "sum", "mean", "amax", "amin", "prod", "logsumexp", "var", "std", "norm",
    "linalg_vector_norm", "all", "any", "count_nonzero",
})


def _is_relevant_fusion(a, b):
    """Is `b(a(x))` a composition worth having as a kernel?

    The epilogue rule, plus one exclusion: an epilogue fused with ANOTHER epilogue is
    usually noise (`abs then relu`), so the first stage must not itself be one -- with
    reductions allowed as a first stage, since `sum then sqrt`-shaped chains are real.
    """
    if b[0] not in FUSION_EPILOGUES:
        return False
    if b[0] in FUSION_INDEX_OUTPUTS or a[0] in FUSION_INDEX_OUTPUTS:
        return False
    # A REDUCTION FED BY A FULL REDUCTION IS DEGENERATE. `std then sum` sums a scalar,
    # which torch computes and the harness then compares as the shaped task it was
    # sized as -- it failed its golden after tracing, emitting and scheduling cleanly.
    # The first stage may be a reduction (`sum then sqrt`-shaped chains are real), but
    # not when the second is one too.
    if a[0] in _FULL_REDUCTIONS and b[0] in _FULL_REDUCTIONS:
        return False
    if a[0] in FUSION_EPILOGUES and a[0] not in (
            "sum", "mean", "amax", "amin", "prod", "logsumexp", "var", "std",
            "norm", "linalg_vector_norm"):
        return False
    return True


def _same_op(a, b, hidx):
    """Are these two rows the SAME operation under different names?

    `abs` and `absolute` decompose identically, so fusing them composes an op with
    itself. Compared by traced-graph signature, the same test the single-op alias
    filter uses -- not by name, which would be a judgement call that goes stale.
    """
    try:
        sa = graph_signature(a[0], a[1], "unary", hidx)
        sb = graph_signature(b[0], b[1], "unary", hidx)
    except Exception:                                  # noqa: BLE001
        return False
    return sa is not None and sa == sb


def _stages_compatible(mod, a, b, nin=1):
    """Does the second stage accept what the first produces?

    Checked by RUNNING the first stage and inspecting the intermediate dtype, rather
    than reasoning about schemas: a bool or integer intermediate feeding a float
    reduction is what made `all -> amax` fail its golden after tracing cleanly.
    """
    import torch as _t
    try:
        xs = tuple(_t.rand(8, 64) + 0.5 for _ in range(nin))
        mid = mod.fns[0](*xs)
        if not _t.is_tensor(mid):
            return False
        if mid.dtype in (_t.bool, _t.int8, _t.int32, _t.int64):
            return False
        out = mod.fns[1](mid)
        return _t.is_tensor(out) and out.numel() > 0 and _t.isfinite(out).all().item()
    except Exception:                                  # noqa: BLE001
        return False


def mine_fused(n_batches, start_batch, per_batch=5, coverage_path=None):
    """Select FUSED kernels: `out = g(f(x))`, both stages from the harvest.

    A KERNEL IS A COMPUTATION, NOT AN OPERATOR. `conv`, `conv + layernorm` and
    `conv + softmax` are three kernels -- three schedules, three fusion opportunities,
    three different right answers for a candidate -- though they compose the same
    registry entries. Selecting only single ops treated them as one, which is why the
    corpus appeared to exhaust PyTorch at 250 while the composition space was untouched.

    PROVENANCE IS UNCHANGED. Every primitive in the fused graph resolves to a harvested
    schema, so `assert_proven` passes exactly as for a single op -- and
    `_would_be_destroyed` is run here too, before a spec is emitted. The composition is
    a new TASK built from harvested ops, not a new operator, and both stages are
    recorded in the selection so a reader can check that.

    MECHANICAL, like the single-op miner: pairs are walked in the harvest's own row
    order, deduplicated by traced-graph signature (a fusion that lowers to a graph
    already in the corpus is the same task), and sized by solving for a tier. Nothing
    is chosen by taste.

    Measured before building: of 120 random ordered pairs of unary expressible ops,
    114 traced and 113 also emitted, scheduled and passed provenance -- 94% viable.
    """
    cov = load_coverage(coverage_path)
    hidx = _harvest_index()
    seen_sigs = corpus_signatures()
    dbg = os.environ.get("MINE_DEBUG")

    unary = []
    for key in cov["covered"]:
        op, ov = key.rsplit(".", 1)
        ov = "" if ov == "default" else ov
        row = hidx.get((op, ov))
        if row is None or not mechanism_eligible(row):
            continue
        sig = C.probe_signature(row)
        if sig not in ("unary", "binary"):
            continue
        unary.append((op, ov, 1 if sig == "unary" else 2))

    # DIAGONAL WALK, not nested loops. `for a: for b:` exhausts every partner of the
    # first op before advancing, so the first 30 picks were all `_pdist_forward -> X`
    # -- a corpus of one op fused with everything, which tests one schedule shape over
    # and over. Walking by OFFSET pairs unary[i] with unary[i+d] for increasing d, so
    # every op appears as a first stage early and the spread is a property of the
    # order rather than of how far the mine happened to run. Still mechanical: the
    # order is the harvest's, and d is just how far apart two rows are in it.
    picked, batches, slot = [], [], 0
    # Only a UNARY op can be a later stage: an epilogue consumes the running value.
    firsts = unary
    seconds = [u for u in unary if u[2] == 1]
    pairs = ((firsts[i], seconds[(i + d) % len(seconds)])
             for d in range(1, max(len(firsts), len(seconds)))
             for i in range(len(firsts)))
    for a, b in pairs:
        if True:  # keeps the body's indentation stable
            if len(picked) >= n_batches * per_batch:
                break
            if a == b:
                continue
            stages = [a[:2], b[:2]]
            from hexkernels.forge.mined import FusedCall
            # FILTER 1 -- THE COMPOSITION MUST COMPUTE SOMETHING NEW.
            # Mechanical pairing produced `abs -> absolute` (which is abs(abs(x))) and
            # `all -> all` (idempotent). Both trace to graphs the signature filter calls
            # distinct, but neither is a distinct COMPUTATION -- it is the alias problem
            # in a new costume, and padding the corpus with it would defeat the reason
            # the alias filter exists at all.
            if a[:2] == b[:2] or a[0] == b[0]:
                continue
            if not _is_relevant_fusion(a[:2], b[:2]):
                continue
            if _same_op(a[:2], b[:2], hidx):
                continue
            mod = FusedCall(stages)
            # FILTER 2 -- THE STAGES MUST BE TYPE-COMPATIBLE.
            # `all -> amax` traced, emitted and scheduled, then FAILED its golden:
            # `all` returns bool and a float reduction over a bool is not a task this
            # pipeline can express. Tracing successfully is not the same as being a
            # valid kernel, so the intermediate dtype is checked before a reference is
            # ever built.
            if not _stages_compatible(mod, a, b, a[2]):
                continue
            preferred = TIER_PATTERN[slot % per_batch]
            got = target = None
            for cand in [preferred] + [t for t in ("T0", "T1", "T2", "T3")
                                       if t != preferred]:
                for shape in SHAPE_LADDER:
                    args = tuple(torch.rand(shape, dtype=torch.float32) + 0.5
                                 for _ in range(a[2]))
                    try:
                        g = trace(mod, args, f"fuse_{a[0]}_{b[0]}")
                    except Exception:                      # noqa: BLE001
                        break                              # this pair never traces
                    if _would_be_destroyed(g, hidx):
                        break
                    if not _golden_is_computable(mod, args):
                        break
                    try:
                        plan = plan_for(g, dtype_bytes=4)
                    except Exception:                      # noqa: BLE001
                        break
                    if plan.tier == cand:
                        got, target = (shape, plan), cand
                        break
                    if plan.tier > cand:
                        break
                if got:
                    break
            if not got:
                continue
            shape, plan = got
            if "hvx" not in plan.mechanisms:
                continue
            sig = tuple(sorted(n.target for n in trace(
                mod, tuple(torch.rand(ALIAS_SHAPE) + 0.5 for _ in range(a[2])),
                "sig").nodes))
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            if dbg:
                print(f"  FUSE   {a[0]} -> {b[0]}  {plan.tier}", file=sys.stderr)
            # AMENDMENT B1: a fused spec is emitted at a DIFFERENT append site
            # from the single-op one, so it needs its own depth/intensity
            # counts. `g` is still the graph from the sizing loop above (the
            # search's last successful trace, at the winning `shape`) -- the
            # re-trace just above rebinds `sig`, not `g`, so this counts the
            # composed graph actually being emitted rather than re-tracing.
            _fused_targets = [n.target for n in g.nodes]
            _n_prim, _n_plumb = _depth_counts(_fused_targets, hidx)
            picked.append({
                # THE FIRST STAGE'S ARITY, not a constant. `mined._args` builds the
                # example tensors from this field, so recording "unary" for a BINARY
                # first stage (`fmax`, `ldexp`, `huber_loss`) handed the module one
                # tensor where it needed two -- every such spec then failed at trace
                # with prims=0, which reads as "the kernel is broken" rather than
                # "we called it with the wrong arity".
                "op": a[0], "overload": "",
                "sig": "unary" if a[2] == 1 else "binary", "synth_plan": None,
                "stages": [[a[0], a[1]], [b[0], b[1]]],
                "dtype": "float32", "dtype_bytes": 4,
                "shape": list(shape), "tier": plan.tier,
                "mechanisms": sorted(plan.mechanisms),
                "working_set": plan.working_set_bytes,
                "schema": f"fused: aten::{a[0]} then aten::{b[0]}",
                "accessor": "",
                "n_primitives": _n_prim,
                "n_plumbing": _n_plumb,
                "ops_per_element": (_n_prim / max(1, math.prod(shape))
                                   if shape else None),
            })
            slot += 1

    # SAME FIX AS `mine()`, same reason: this loop used to drop a trailing
    # partial group instead of keeping it, discarding already-selected valid
    # fused picks with no record at all (mine_fused never even exposed a
    # `.remainder`). Selection order and dedup, both above, decide which
    # fused picks exist; grouping them into batches must not be a second,
    # silent selection step.
    for i in range(0, len(picked), per_batch):
        batches.append(picked[i:i + per_batch])
    return batches


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batches", default="16-25", help="inclusive range, e.g. 16-25")
    ap.add_argument("--per-batch", type=int, default=5)
    ap.add_argument("--json", default=str(DEFAULT_POOL))
    ap.add_argument("--append", action="store_true",
                    help="extend an existing selection instead of replacing it")
    ap.add_argument("--coverage", default=None,
                    help="coverage JSON to mine from; defaults to the plain cache. "
                         "Pass run_artifacts/forge2/coverage_synth.json with --synth")
    ap.add_argument("--fuse", action="store_true",
                    help="mine FUSED kernels (g(f(x))) instead of single ops -- a "
                         "kernel is a computation, not an operator, and the single-op "
                         "space is exhausted at 250")
    ap.add_argument("--synth", action="store_true",
                    help="also mine rows reachable only with SYNTHESISED argument "
                         "values (coverage_synth.json's covered_synth) -- attention, "
                         "the RNN cells, the margin losses. Each such entry records "
                         "the argument plan it was built from")
    args = ap.parse_args(argv)

    lo, hi = (int(x) for x in args.batches.split("-"))
    n = hi - lo + 1
    if args.fuse:
        batches = mine_fused(n, lo, args.per_batch, coverage_path=args.coverage)
    else:
        batches = mine(n, lo, args.per_batch, synth=args.synth,
                       coverage_path=args.coverage)

    # --append EXISTS BECAUSE THE DEFAULT IS DESTRUCTIVE AND THE FILE IS THE CORPUS.
    #
    # `mined.py` builds batches 16+ from this one file at import, so writing it
    # fresh for `--batches 26-40` does not add ten batches -- it DELETES batches
    # 16-25 and renumbers the new ones on top of them. Fifty verified kernels would
    # stop existing as specs while their artifacts and candidates stayed on disk,
    # which is worse than an error because everything still runs.
    #
    # Appending is also the honest shape: one committed selection describing the
    # whole mined series in the harvest's own order, so a diff to the file is a diff
    # to the corpus. The alternative -- a second file plus a loader that reads both
    # -- makes "what is in the corpus" a question with two places to look.
    path = pathlib.Path(args.json)
    if args.append and path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        first, existing = doc["first_batch"], doc["batches"]
        expected = first + len(existing)
        if lo != expected:
            raise SystemExit(
                f"--append needs --batches to start at {expected} (first_batch "
                f"{first} + {len(existing)} existing batches), got {lo}. Appending "
                "at any other number would renumber batches that already have "
                "verified kernels and artifacts on disk.")
        if args.per_batch != doc["per_batch"]:
            raise SystemExit(
                f"--append needs --per-batch {doc['per_batch']} to match the "
                f"existing selection, got {args.per_batch}")
        out = {"first_batch": first, "per_batch": doc["per_batch"],
               "batches": existing + batches}
        print(f"appending {len(batches)} batches after {len(existing)} existing "
              f"(batches {lo}-{lo + len(batches) - 1})")
    else:
        out = {"first_batch": lo, "per_batch": args.per_batch, "batches": batches}
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")

    print(f"mined {len(batches)} batches of {args.per_batch} "
          f"({sum(len(b) for b in batches)} kernels)")
    for bi, chunk in enumerate(batches):
        print(f"\nbatch {lo + bi}:")
        for k in chunk:
            print(f"  {k['op']}.{k['overload'] or 'default':<14s} "
                  f"{k['dtype']:<8s} {str(tuple(k['shape'])):<14s} "
                  f"{k['tier']}  {','.join(k['mechanisms'])}")
    if len(batches) < n:
        print(f"\nSHORT: asked for {n} batches, mined {len(batches)}. The covered "
              f"list ran out of ops that (a) are not already in the corpus, "
              f"(b) survive provenance, and (c) can reach their slot's tier.")
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
