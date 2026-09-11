"""Check the hand-derived schedule against real Linalg IR from torch-mlir.

WHY THIS IS A SEPARATE SCRIPT AND NOT A TEST
--------------------------------------------
`hexkernels.forge.frontend.schedule` derives `iterator_types` and indexing maps
from a hand-written table, one entry per primitive. That table is the pipeline's
only statement of what each op's loop nest is, and nothing else can catch it
being wrong -- a bad entry produces a kernel that compiles, runs, and is subtly
misscheduled. torch-mlir lowers the same graph to Linalg, which carries exactly
those two facts, so it is ground truth for the table.

It is a script rather than a gate test because torch-mlir is a 71 MB binary
wheel published as a GitHub release asset, not on PyPI. Requiring it would make
`pytest -q -m "not slow"` depend on a download. Run this when you touch the
table or add a primitive -- which is when it can silently go wrong.

INSTALL (one-off; ~71 MB, no torch pin, coexists with this repo's torch)
-----------------------------------------------------------------------
The PyPI package is abandoned -- its newest release is 20221213.686 from
December 2022, with cp310/cp37 wheels only, so `pip install torch-mlir` fails on
the Python version and reports "no matching distribution". **That failure means
"wrong Python", not "unavailable", and reading it as the latter cost this project
a wrong design decision.** Current wheels live here:

    https://github.com/llvm/torch-mlir-release/releases/tag/dev-wheels

    pip install --no-deps <URL of torch_mlir-<date>-cp311-cp311-win_amd64.whl>

`--no-deps` matters: the wheel declares only numpy and packaging, and it imports
cleanly against this repo's torch 2.7.1. Pin an exact dated wheel -- these are
nightlies and the API has already moved once (the pre-2024 `torch_mlir.compile`
is gone; the current entry point is `torch_mlir.fx.export_and_import`).

WHAT IS COMPARED
----------------
Iterator types AND indexing maps -- the maps being the half that was never
checked before 2026-08-03, and the half where a schedule has room to be wrong.
Named ops are generalised first (`linalg.generalize`), because otherwise the two
ops with non-trivial nests (`linalg.matmul`, `linalg.conv_2d_nchw_fchw`) are
exactly the two whose schedule lives in a C++ op definition rather than the
printed IR -- so the script was silent on the only cases worth checking.

`EXACT`         nests and maps agree as printed.
`canonical`     agree after renaming loops so the result map is the identity and
                dropping our extent-1 REDUCTION loops -- both provably
                access-pattern-preserving. The depthwise-convolution case:
                `aten.convolution` keeps a singleton input-channel axis on a
                `(512, 1, 3, 3)` weight so our nest has 7 loops, while torch-mlir
                collapses the weight to `(512, 3, 3)` for
                `linalg.depthwise_conv_2d_nchw_chw` and has 6, with the channel
                loop last. A SECOND pass only -- a raw agreement is still EXACT,
                so this can turn a DIFFER into an explained match and never the
                other way. Negative cases pinned in `tests/test_validate_schedule.py`:
                a wrong stride coefficient, a transposed operand and a missing
                reduction all still report DIFFER.
`extra-result`  same nest, same maps, plus a surplus operand whose map DUPLICATES
                one already present. The fused-argmax case: torch-mlir routes
                softmax's max through `aten.max.dim` (values + indices) while our
                decomposition gives `aten.amax` (values only), so its generic has
                a third operand mapped `(d0)`, like the value result. A surplus
                map that is *new* is not this -- it reports DIFFER.
`fused-into-reduce`
                the compiler folded an elementwise PRODUCER into the body of its
                reduction CONSUMER, leaving no parallel nest at all. Distinct from
                `fused`, which requires a surviving parallel nest of the same
                signature to fuse into. The `fp32_all` case: `all(x)` decomposes to
                logical_not -> any.dims -> logical_not, and torch-mlir emits ONE
                `R,R` generic whose body is `cmpf une` then `andi`. Accepted only
                when the reduction nest has the same loop count as the surplus
                parallel nest and carries that nest's identity map among its own
                operands -- so the reduction provably reads an operand with the
                access pattern of the pass that disappeared.
`addressing-in-body`
                same access pattern, written where an indexing-map comparison
                cannot read it. Two mechanisms, both measured on batches 36-43:
                `linalg.index` puts the loop index in the generic's BODY (`tril`
                collapses arange -> unsqueeze -> sub -> le into one identity nest;
                `diag` and `trace` do the diagonal as `linalg.index 0` plus a scalar
                `tensor.extract`), and `tensor.extract_slice` puts an offset in a
                VIEW (`glu`'s two halves, after which both generics are identity).
                Ours spells the same access in the map -- `(d0) -> (d0, d0)`,
                `(d0, d1) -> (d0, d1 + 512)`. Available ONLY when the compiler's own
                IR carries the token, and never over a TRANSPOSE: an index in the
                body explains an offset or a gather, and must not become a blanket
                excuse for any surplus in any kernel whose body computes an index.
`DIFFER`        a real disagreement. Investigate before shipping the kernel.

RANK-0 OPS ARE SET ASIDE, ON BOTH SIDES. An op with no loops and only `() -> ()`
maps has no iteration space, so it has no access pattern to agree or disagree
about: `all`'s trailing scalar `logical_not` on our side, and the generalised
`linalg.fill` that seeds an accumulator on the compiler's. Every verdict names
what it set aside, so this is visible rather than silent. Doing it one-sidedly
does not work and was tried: filtering only ours left the compiler's rank-0 fill
in `par_g`, which moved the disagreement instead of resolving it.

Measured 2026-08-03 over batches 1-3, all 15 kernels:
**10 EXACT, 1 canonical, 4 extra-result, 0 DIFFER.**

Measured 2026-08-05 over the WHOLE corpus, batches 1-43, all 215 kernels:
**104 EXACT, 2 canonical, 9 extra-result, 16 fused, 2 staged-reduce, 5 scan,
3 multi-pass, 1 fused-operand, 2 fused-into-reduce, 5 addressing-in-body,
60 no-linalg-form, 2 outside-linalg, 1 DIFFER.**

THE ONE DIFFER IS REAL AND IS NOT A TABLE ERROR: `fp32_cosine_similarity`. The two
frontends DECOMPOSE THE ATEN OP DIFFERENTLY, and the difference is substantive
rather than bookkeeping:

  ours      norm(x) -> [32,1], norm(y) -> [32,1], clamp both, divide x and y at
            RANK 2 (4,096 divisions), multiply, sum
  compiler  multiply, sum -> [32], norm(x) -> [32], norm(y) -> [32], multiply the
            norms, divide ONCE at RANK 1 (32 divisions)

Both compute cosine similarity and both are correct; the compiler's shape does 128x
fewer divisions. Our table faithfully annotates the graph `torch.export` handed it,
so there is no entry to fix -- the disagreement is one layer up, between two
frontends' decompositions of `aten.cosine_similarity`. Left as a DIFFER on purpose:
it is the instrument reporting something true, and a verdict invented to absorb a
single case would be a rule fitted to its only example. The kernel itself is
verified correct against its own generated golden.

Batches 36-43 raised 15 DIFFERs, and they came from THREE causes that one rule could
not have separated. The first attempt did try one rule -- "accept any projection" --
and it turned `test_a_surplus_nest_with_a_DIFFERENT_map_is_still_DIFFER` green,
which is how the causes were told apart rather than lumped:
  * 7 were a RANK-0 OPERAND (a `scalar_tensor` feeding a `where`), now folded into
    `fused` by `_identity_but_for_scalars` -- the module's existing rank-0 judgement
    applied at operand granularity instead of nest granularity.
  * 7 were the compiler moving addressing out of the maps -> `addressing-in-body`.
  * 1 was `cosine_similarity`, above, which the loose rule would have hidden.

The `no-linalg-form` count grew with the corpus, not as a fraction of it: batches
26-35 are the transcendental tail and the `.Scalar` overloads, and torch-mlir
declines to lower `sin`, `tan`, `sinh`, `logaddexp2`, `nextafter`, `signbit` and
most of `special_*` at all. That is a limit of the compiler reached before any of
this pipeline's code runs, and `test_every_batch_kernel_gets_both_irs` bounds it
with a counted, attributed ceiling rather than treating it as a pass.

That is the first run to cover the whole corpus, and covering it was a fix:
`_all_batches()` iterated `kernels.BATCHES` (since renamed
`HANDWRITTEN_BATCHES`, for exactly this reason), which registers only the
hand-written 1-15, so this script's default scope had silently been 15 batches
out of 25 -- printing a clean summary for the two thirds it had looked at. It now
calls `kernels.all_batches()`. The 19 `no-linalg-form` rows are mined ops
torch-mlir declines to lower at all (`logical_not` on an integer, `hypot`,
`isclose`, `lgamma`, `_weight_norm` and others); that is an absence with a named
cause, not a disagreement.

THIS SCRIPT HAS PAID FOR ITSELF THREE TIMES
-------------------------------------------
* Our `keepdim` reduction result was rendered `(d0, d1) -> (d0)` where Linalg
  emits `(d0, d1) -> (d0, 0)`.
* Comparing by a dict keyed on iterator signature silently discarded one of
  softmax's two `P,R` nests -- switching to a multiset is what exposed the
  fused-argmax difference above, which had been hidden rather than absent.
* `linalg.summary`'s alias regex terminated on `[^>]*>`, which stops inside the
  map's own `->`, so every multi-dimensional map had been truncated to
  `affine_map<(d0, d1) ->`. The test guarding it asserted only
  `startswith("affine_map<")`, which a truncated string satisfies.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap


def _all_batches():
    from hexkernels.forge.kernels import all_batches
    return all_batches()

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Runs inside the torch-mlir interpreter, which may be a different environment
# from the one importing this module -- hence a subprocess and a JSON handoff
# rather than an import.
_DUMP = textwrap.dedent('''
    import json, sys, torch
    sys.path.insert(0, r"{repo}")
    from hexkernels.forge import linalg as L
    from hexkernels.forge.kernels import batch

    out = {{}}
    for b in {batches}:
        for spec in batch(b):
            try:
                # generalized=True: named ops carry their schedule in the C++ op
                # definition, not the printed IR, so without this pass the two
                # ops with non-trivial nests are exactly the two that cannot be
                # checked. See linalg.generalize.
                ir = L.linalg_ir(spec.module, spec.args, generalized=True)
                s = L.summary(ir)
                import re as _re
                out[spec.name] = {{
                    "ops": [{{"iter": g["iterator_types"],
                             "maps": g["indexing_maps"]}}
                            for g in s["generics"]],
                    "named": s["named_ops"],
                    # WHICH NON-LINALG STRUCTURED OPS THE COMPILER CHOSE. This is
                    # the evidence behind the `outside-linalg` verdict: when
                    # torch-mlir routes an op to `tm_tensor.scan` or
                    # `tm_tensor.sort` there is no linalg.generic to compare a
                    # reduction nest against, and saying so requires knowing that
                    # it happened rather than inferring it from an absence.
                    "outside": sorted(set(
                        _re.findall(r"tm_tensor\\.[a-z_0-9]+", ir))),
                    # WHERE THE COMPILER PUT ADDRESS ARITHMETIC WHEN IT DID NOT PUT
                    # IT IN AN INDEXING MAP. Same role as `outside` above: evidence
                    # for a verdict, collected here because this is the only place
                    # the FULL IR exists -- `L.summary` keeps generics, so an
                    # `extract_slice` that DEFINES a generic's operand is dropped
                    # before the parent ever sees it. Inferring either token from an
                    # absence of maps would make the verdict unfalsifiable.
                    "addressing": sorted(set(
                        _re.findall(r"linalg\\.index|tensor\\.extract_slice", ir))),
                }}
            except Exception as e:
                out[spec.name] = {{"error": "{{}}: {{}}".format(
                    type(e).__name__, str(e)[:200])}}
    print(json.dumps(out))
''')


def real_linalg(python_exe, batches) -> dict:
    """Lower every kernel in `batches` to Linalg using `python_exe`."""
    src = _DUMP.format(repo=REPO, batches=tuple(batches))
    r = subprocess.run([python_exe, "-c", src], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"torch-mlir dump failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout)


def derived(batches) -> dict:
    """Our own annotation for the same kernels, in the shape `real_linalg` returns.

    `extents` is carried alongside because it is what licenses dropping a
    degenerate loop during canonicalisation, and the extent is not recoverable
    from the map text.
    """
    from hexkernels.forge.frontend.schedule import annotate
    from hexkernels.forge.frontend.trace import trace
    from hexkernels.forge.kernels import batch as _b
    out = {}
    for b in batches:
        for spec in _b(b):
            scheds = annotate(trace(spec.module, spec.args, spec.name))
            out[spec.name] = {"ops": [
                {"iter": list(s.iterator_types),
                 "target": s.target,
                 "extents": [l.extent for l in s.loops],
                 "maps": [o.affine_map(s.loops) for o in s.operands]}
                for s in scheds]}
    return out


# Ops whose emitted body moves a value without computing anything: a cast, a
# clone, an order-preserving reshape. Sourced from the emitter tables rather than
# listed by hand, so a new copy-like row cannot fall outside it.
def _value_moving() -> frozenset:
    from hexkernels.forge.frontend.primitives import ELEMENTWISE, FLAT_COPY
    movers = {k for k, tmpl in ELEMENTWISE.items() if tmpl.strip() == "$0"}
    return frozenset(movers | set(FLAT_COPY))


VALUE_MOVING = _value_moving()


def _identity_maps(op) -> bool:
    """Every operand indexed by the full loop nest IN ORDER -- an elementwise pass
    with no broadcast, no transpose and no reduced axis.

    This used to return `len(set(maps)) == 1`, which is "all operands share one map"
    and NOT what the docstring says. A nest whose every operand is transposed the
    same way -- `(d0, d1) -> (d1, d0)` twice -- satisfies it, so a transposed pass
    counted as an identity pass. That was harmless while every caller also compared
    the map against the compiler's own operands, and became a hole the moment the
    `fused-into-reduce` rule stopped doing so.

    Now it checks the thing: each map's results must be exactly the loop variables in
    declaration order.
    """
    maps = op.get("maps") or []
    if not maps or len(set(maps)) != 1:
        return False
    body = maps[0].split("->", 1)
    if len(body) != 2:
        return False
    dims = re.findall(r"d\d+", body[0])
    results = [r.strip() for r in body[1].strip().strip("<>() ").split(",")]
    return bool(dims) and results == dims


def _identity_but_for_scalars(op) -> bool:
    """`_identity_maps`, except that a RANK-0 OPERAND MAP is ignored.

    THE SAME JUDGEMENT THIS MODULE ALREADY MAKES ABOUT RANK-0 NESTS, at the operand
    granularity. A rank-0 nest is set aside from the surplus comparison because it
    has no iteration space and so "no access pattern for it to disagree about" (see
    the `_rank0` block in `compare`). An operand read through `(d0, d1) -> ()` is the
    same fact one level down: it is a SCALAR, the same element for every iteration of
    the nest, and it can therefore encode no schedule -- no stride, no order, no
    tiling. The compiler does not materialise a tensor for it at all; the value
    becomes an `arith.constant` in the generic's body.

    Measured: this is 7 of the 15 DIFFERs in batches 36-43, and every one has the
    same shape -- `hardshrink`, `heaviside`, `kl_div`, `logit`, `log_sigmoid`,
    `lerp.Scalar` and `binary_cross_entropy_with_logits` all decompose to a
    `scalar_tensor`/`full` feeding a `where` or a compare, so our table records a
    nest whose surplus operand map is `(d0, d1) -> ()`. The rank-0 PRODUCER was
    already set aside; the CONSUMER's rank-0 operand was not, so the kernel was
    reported as disagreeing with the compiler over a constant.

    NARROW ON PURPOSE. A `(d0, d1) -> (d0, 0)` broadcast is NOT ignored: a size-1
    axis is still an axis, the operand is a real tensor, and which element each
    iteration reads is a genuine question about it. That is the distinction
    `test_a_surplus_nest_with_a_DIFFERENT_map_is_still_DIFFER` pins, and relaxing
    this to "any projection" turned that test green -- which is how the difference
    between the two cases got noticed rather than assumed.
    """
    # NOT `_map_loops`, which scans the WHOLE map text -- declaration included -- so
    # `(d0, d1) -> ()` reports {d0, d1} and no rank-0 operand would ever be filtered.
    # The rank-0-ness of an operand is a property of the map's RESULT.
    def _rank0_result(m):
        body = m.split("->", 1)
        return len(body) == 2 and not body[1].strip().strip("<>() ")

    maps = [m for m in (op.get("maps") or []) if not _rank0_result(m)]
    if not maps:
        return False
    return _identity_maps({"maps": maps})


def _is_transpose(op) -> bool:
    """True when any operand map REORDERS the loops it keeps.

    The one access-pattern difference that no rule in this module may excuse. A
    broadcast drops or pins an axis and a gather repeats one, and both can be
    accounted for by something the compiler did elsewhere; a transpose means the two
    schedules walk memory in a different ORDER, which is a real disagreement about
    the loop nest and the thing `_identity_maps` was originally guarding.

    Kept separate from `_addressing_outside_maps` so the `addressing-in-body` rule
    has a floor: `linalg.index` in the IR explains an offset or a diagonal, and it
    must not become a blanket excuse for any surplus in any kernel whose body happens
    to compute an index.
    """
    for m in op.get("maps") or []:
        body = m.split("->", 1)
        if len(body) != 2:
            continue
        dims = re.findall(r"d\d+", body[0])
        rhs = body[1].strip().strip("<>() ")
        kept = [t.strip() for t in rhs.split(",") if t.strip().startswith("d")]             if rhs else []
        # DEDUPE FIRST. A REPEATED loop -- `(d0) -> (d0, d0)`, the diagonal -- is a
        # GATHER, not a reorder, and comparing the raw list against the declaration
        # order calls it a transpose because ['d0', 'd0'] != ['d0']. That false
        # positive sent `diag` and `trace` back to DIFFER after this floor was added,
        # which is how it was caught. The gather is handled by
        # `_addressing_outside_maps` on the compiler's `linalg.index` evidence; this
        # function's whole job is ORDER.
        seen, uniq = set(), []
        for k in kept:
            if k in dims and k not in seen:
                seen.add(k)
                uniq.append(k)
        if uniq != [d for d in dims if d in seen]:
            return True
    return False


def _addressing_outside_maps(op) -> bool:
    """True when `op`'s surplus maps carry address arithmetic that is NOT a
    projection -- an offset, a scale, or a repeated loop.

    Paired with the `addressing` evidence from the compiler: this says our side
    spells an access pattern in the map, and `addressing` says the compiler spelled
    the same kind of thing in the body or in a view. Both are required.
    """
    for m in op.get("maps") or []:
        body = m.split("->", 1)
        if len(body) != 2:
            continue
        rhs = body[1].strip().strip("<>() ")
        terms = [t.strip() for t in rhs.split(",")] if rhs else []
        kept = [t for t in terms if t.startswith("d")]
        if any(("+" in t or "*" in t) for t in terms):
            return True
        if len(kept) != len(set(kept)):
            return True
    return False


def _addressing_note(ir_tokens) -> list:
    """Tokens showing the compiler put address arithmetic somewhere this comparison
    cannot see it -- in the loop BODY or in a VIEW rather than in an indexing map.

    Both are real and both were found in batches 36-43:

      * `linalg.index` -- the compiler computes the loop index inside the generic's
        body. `tril` collapses arange -> unsqueeze -> sub -> le into ONE identity
        `P,P` generic whose body calls `linalg.index 0` and `linalg.index 1`;
        `diag` and `trace` do the diagonal gather as `linalg.index 0` plus a scalar
        `tensor.extract` at (i, i). Our table spells the same access in the map, as
        `(d0) -> (d0, d0)`.
      * `tensor.extract_slice` -- the offset lives in a view op, not a generic.
        `glu`'s two halves arrive as `%extracted_slice` and `%extracted_slice_0`,
        after which both generics are plain identity; our table carries the offset as
        `(d0, d1) -> (d0, d1 + 512)`.

    A comparison of iterator signatures and indexing maps is blind to both, so the
    resulting DIFFER says "I cannot see it", not "the compiler disagrees". This
    returns the evidence rather than assuming it: the verdict is only available when
    the compiler's own IR contains the token, so a kernel whose table really does
    invent an access pattern still reports DIFFER.
    """
    return sorted(ir_tokens or ())


def _surplus_ops(ops, par_m, par_g) -> list:
    """The all-parallel ops of `ops` that have no counterpart in linalg, by
    signature multiplicity."""
    need = {}
    for sig in par_g:
        need[sig] = need.get(sig, 0) + 1
    out = []
    for o in ops:
        if "reduction" in o["iter"]:
            continue
        sig = _sig(o)
        if need.get(sig, 0) > 0:
            need[sig] -= 1
        else:
            out.append(o)
    return out


def _sig(op) -> str:
    return ",".join(t[0].upper() for t in op["iter"])


# --- canonicalisation ------------------------------------------------------
#
# Two schedules can describe the same access pattern in different words, and both
# differences below are provably pattern-preserving rather than judgement calls.
# Everything else is left to report as a disagreement.
#
#   1. LOOP NAMING. Nothing forces two lowerings to number the loops alike.
#      torch-mlir's `depthwise_conv_2d_nchw_chw` puts the channel loop LAST
#      (result `(d0, d3, d1, d2)`) where we put it second (`(d0, d1, d2, d3)`).
#      Renaming both so the result map becomes the identity removes the choice.
#
#   2. EXTENT-1 REDUCTION LOOPS. A reduction that runs once accumulates nothing
#      and can be neither reordered nor tiled. `aten.convolution` keeps a
#      singleton input-channel axis on a depthwise weight `(512, 1, 3, 3)`, so our
#      nest has a 7th loop of extent 1; torch-mlir collapses that weight to
#      `(512, 3, 3)` and has 6 loops. Dropping ours also turns the weight's
#      singleton axis into a constant, which is how the collapse falls out.
#
# Only OUR side is drop-normalised, and deliberately: the point is to remove a
# singleton we retain, not to tolerate one in the compiler's output. If a future
# lowering keeps a degenerate loop we do not, that still reports DIFFER.

_TERM = re.compile(r"(?:(\d+)\s*\*\s*)?d(\d+)|d(\d+)\s*\*\s*(\d+)")


def _degenerate(op) -> set:
    """Loop indices that are REDUCTIONS of extent 1.

    Reductions only. An extent-1 *parallel* loop is equally degenerate in
    principle, but both sides keep theirs (a batch axis of 1 stays a loop in both
    lowerings), so dropping ours would desynchronise the comparison rather than
    align it -- and it silently shortened the result map's rank, which is how this
    was caught.
    """
    return {k for k, (kind, ext) in enumerate(zip(op["iter"],
                                                  op.get("extents") or []))
            if ext == 1 and kind == "reduction"}


def _parse_map(m: str):
    """`affine_map<(..) -> (d0, d1 + d4 * 2, 0)>` -> per-axis [(loop, coeff)]."""
    body = m.split("->", 1)[1] if "->" in m else m
    axes = []
    for res in body.strip().strip("<>() ").split(","):
        terms = []
        for part in res.split("+"):
            part = part.strip()
            if not part or part == "0":
                continue
            mm = _TERM.fullmatch(part)
            if mm is None:               # something we do not model; bail out
                return None
            if mm.group(2) is not None:
                terms.append((int(mm.group(2)), int(mm.group(1) or 1)))
            else:
                terms.append((int(mm.group(3)), int(mm.group(4))))
        axes.append(tuple(sorted(terms)))
    return axes


def _canon(op, drop_extent_1: bool = False):
    """Canonical form of one op's maps: loops renamed so the result map is the
    identity, optionally with extent-1 loops removed.

    Returns None when the maps are not in the modelled affine form, so the caller
    falls back to raw text comparison rather than silently claiming a match.
    """
    parsed = [_parse_map(m) for m in op["maps"]]
    if any(p is None for p in parsed):
        return None
    if drop_extent_1:
        degenerate = _degenerate(op)
        parsed = [[tuple(t for t in ax if t[0] not in degenerate) for ax in p]
                  for p in parsed]
    else:
        degenerate = set()
    result = parsed[-1]            # Linalg convention: `outs` is the last map
    rename, nxt = {}, 0
    for ax in result:              # result axes fix the leading loop order
        if len(ax) == 1 and ax[0][1] == 1 and ax[0][0] not in rename:
            rename[ax[0][0]] = nxt
            nxt += 1
    used = sorted({t[0] for p in parsed for ax in p for t in ax})
    for loop in used:              # remaining (reduction) loops, in index order
        if loop not in rename:
            rename[loop] = nxt
            nxt += 1
    def render(p):
        out = []
        for ax in p:
            if not ax:
                continue           # a constant axis; carries no stride
            renamed = sorted((rename[loop], coeff) for loop, coeff in ax)
            out.append(" + ".join(f"e{e}" if c == 1 else f"e{e} * {c}"
                                  for e, c in renamed))
        return "(" + ", ".join(out) + ")"
    return tuple(sorted(render(p) for p in parsed))


def _canon_sig(op, drop_extent_1: bool = False) -> str:
    """Iterator signature with extent-1 loops optionally removed, order-insensitive.

    Sorted, because canonicalising the loop NAMES makes positional order
    meaningless -- what survives is how many loops of each kind the nest has.
    """
    drop = _degenerate(op) if drop_extent_1 else set()
    kinds = [k for i, k in enumerate(op["iter"]) if i not in drop]
    return ",".join(sorted(t[0].upper() for t in kinds))


def _norm_map(m: str) -> str:
    """Whitespace-normalised, with constant results dropped.

    Dropping `0` results is the one licensed normalisation, and it exists for a
    real decomposition difference rather than to make a comparison pass. Measured
    on softmax: torch-mlir's `sum` emits `(d0, 0)`, the same rank-2 form we do,
    but its `max` goes through `aten.max.dim`, which produces a RANK-1 result and
    a separate `expand` -- so that one op alone writes `(d0)` where we write
    `(d0, 0)`. A constant result contributes element stride 0 (the axis does not
    advance), so the two describe the same access pattern, and comparing the raw
    text would report a difference that is purely about which decomposition ran.

    Deletion-checked 2026-08-03: without this, both softmax kernels report DIFFER
    and nothing else changes -- so the normalisation is doing exactly this job and
    is not masking anything broader.
    """
    body = m.split("->", 1)[1] if "->" in m else m
    results = [r.strip() for r in body.strip().strip("<>() ").split(",")]
    return "(" + ", ".join(" ".join(r.split()) for r in results if r != "0") + ")"


def _extra_dups(ours, theirs):
    """`theirs` minus `ours`, but only if every surplus map DUPLICATES one we
    already have. Returns the surplus list, or None if not that situation.

    This is the fused-argmax case and it is narrow on purpose. torch-mlir routes
    softmax's max through `aten.max.dim`, which returns values AND indices, so its
    generic has a third operand -- the i64 index tensor -- mapped `(d0)`, exactly
    like the value result. PyTorch's own decomposition gives us `aten.amax`, values
    only. Same loop nest, same access patterns; one extra *result* that a different
    decomposition happened to compute alongside.

    A surplus map that is NOT already present would mean Linalg touches memory in
    a pattern our schedule does not describe -- a real disagreement, and this
    returns None so it is reported as one.
    """
    rest = list(theirs)
    for m in ours:
        if m not in rest:
            return None
        rest.remove(m)
    return rest if rest and all(x in ours for x in rest) else None


from hexkernels.forge.frontend.primitives import (  # noqa: E402
    MULTI_PASS_REDUCTIONS as _MULTI_PASS)


def _map_loops(map_text: str) -> set:
    """The loop names an affine map mentions. `(d0, d2)` -> {"d0", "d2"}.

    Used to check that a surplus operand map introduces NO NEW LOOP, which is
    what separates a fused broadcast from a different access pattern.
    """
    return set(re.findall(r"d\d+", map_text))


def compare(mine, real) -> list:
    """Row per kernel. Reduction-bearing ops are compared strictly, both types
    and MAPS; all-parallel ops are required to be a SUBSET of the compiler's.

    Why the asymmetry: no plumbing op has a reduction loop, so a reduction
    signature we derive must appear in Linalg and vice versa. All-parallel
    generics, though, include Linalg's own plumbing -- the `linalg.fill` that
    establishes a convolution accumulator, and the f32 -> f16 truncation it
    inserts after one -- which our graph legitimately does not have. Requiring
    equality there would report those as disagreements; requiring containment
    still catches an all-parallel schedule we invented.

    Maps are compared only on reduction-bearing ops. Those are where a schedule
    has degrees of freedom to get wrong (matmul's transposed B operand,
    convolution's `d2*stride + d5*dilation`), and they avoid the broadcast
    rank difference that `_norm_map` documents.
    """
    rows = []
    for name, m in mine.items():
        r = real.get(name, {})
        # SCAN NESTS ARE OUTSIDE LINALG'S VOCABULARY, and that is a verdict rather
        # than a disagreement.
        #
        # `linalg.generic` carries exactly two iterator kinds, parallel and
        # reduction. A scan is neither -- iteration k reads what k-1 wrote, and the
        # axis keeps its full extent -- so there is no Linalg spelling for the nest
        # to be compared AGAINST. torch-mlir agrees in the strongest possible way:
        # it routes `cumsum` to `tm_tensor.scan`, a different dialect, and for
        # `cummax` it refuses to legalize at all.
        #
        # Reported as its own verdict, carrying WHAT THE COMPILER ACTUALLY DID, so
        # this cannot become a way to make an inconvenient kernel stop being
        # checked: a scan nest of ours that the compiler HAD expressed as a
        # linalg.generic would show up here as a structured-op count to explain.
        scan_ops = [o for o in m["ops"] if "scan" in o["iter"]]
        other_ops = [o for o in m["ops"] if "scan" not in o["iter"]]
        if scan_ops and not other_ops:
            if "error" in r:
                why = ("torch-mlir did not lower this graph at all: "
                       + r["error"].strip().splitlines()[0][:160])
            else:
                why = (f"torch-mlir lowered it outside linalg -- "
                       f"{len(r.get('ops', []))} structured op(s) for "
                       f"{len(scan_ops)} scan nest(s)")
            rows.append((name, "scan",
                         "linalg has two iterator kinds and cannot express a "
                         "scan; " + why))
            continue
        if "error" in r:
            # THE COMPILER PRODUCED NO IR TO COMPARE AGAINST, which is not a
            # disagreement -- there is nothing to disagree with. torch-mlir marks
            # some ATen ops illegal outright: `aten.cummax` (batch 8) and
            # `aten.median.dim` (batch 11) both fail to legalize. Reported as its
            # own verdict carrying the compiler's own diagnostic, and NARROW on
            # purpose: it fires only when the lowering FAILED. A lowering that
            # succeeded and disagreed still reports DIFFER, so this cannot become
            # a way to make an inconvenient comparison disappear.
            rows.append((name, "no-linalg",
                         "torch-mlir cannot lower this op, so there is no "
                         "structured IR to compare: "
                         + r["error"].strip().splitlines()[0][:150]))
            continue
        notes = []
        if scan_ops:
            notes.append(f"{len(scan_ops)} scan nest(s) have no linalg counterpart "
                         "and were not compared")
        # A sorted MULTISET of (signature, maps), not a dict keyed by signature:
        # softmax's amax and sum share the signature `P,R`, so a dict keyed by it
        # would keep one and silently discard the other's maps. Switching to a
        # multiset is what surfaced the fused-argmax difference below, which the
        # dict form had been hiding.
        red = lambda ops: sorted(
            (_sig(o), tuple(sorted(_norm_map(x) for x in o["maps"])))
            for o in ops if "reduction" in o["iter"])
        red_m, red_g = red(other_ops), red(r["ops"])
        unmatched, pool, extras = [], list(red_g), []
        for sig, maps in red_m:
            if (sig, maps) in pool:
                pool.remove((sig, maps))
                continue
            hit = next((c for c in pool if c[0] == sig and _extra_dups(maps, c[1])),
                       None)
            if hit is None:
                unmatched.append((sig, maps))
            else:
                pool.remove(hit)
                extras.append(f"[{sig}] +{_extra_dups(maps, hit[1])}")
        # Anything still unmatched gets a second pass in canonical form: loops
        # renamed so the result map is the identity, and our extent-1 reduction
        # loops dropped. Both are pattern-preserving (see the canonicalisation
        # note). This is a SECOND pass, not a replacement -- a kernel that matched
        # on raw text is still reported EXACT, so canonicalisation can only turn a
        # DIFFER into an explained match, never hide a raw disagreement.
        canonical = []
        if unmatched and pool:
            cm = lambda ops, drop: sorted(
                (_canon_sig(o, drop), _canon(o, drop))
                for o in ops if "reduction" in o["iter"])
            can_m, can_g = cm(m["ops"], True), cm(r["ops"], False)
            if all(c is not None for _, c in can_m + can_g) and can_m == can_g:
                unmatched, pool = [], []
                canonical = [s for s, _ in can_m]
        # THE COMPILER PUT THIS OP IN A DIFFERENT DIALECT, so there is no linalg
        # reduction nest to compare against -- not a disagreement, an absence with
        # a named cause. Measured: `sort` and `topk` both lower to
        # `tm_tensor.sort`, leaving only an all-parallel index-materialising
        # generic behind, which is why `red_g` is empty while ours is not.
        #
        # Attributed to the COMPILER'S OWN output: it fires only when the dump
        # reports a `tm_tensor.*` op. Our table's opinion is not evidence for a
        # claim about the compiler, so an empty `red_g` with no such op is still a
        # DIFFER.
        outside = []
        if red_m and not red_g and r.get("outside"):
            outside = list(r["outside"])
            unmatched, pool = [], []
        staged, multipass, fused_op, into_red = [], [], [], []
        if not outside and (unmatched or pool):
            # A MULTI-AXIS REDUCTION THE COMPILER STAGES AS A CHAIN. Measured on
            # `fp32_amax_all`: ours is one nest reducing both axes ("R,R"), linalg is
            # two ops -- reduce the last axis ("P,R") then reduce the result ("R").
            # Same computation, same total accumulation, different staging, and the
            # emitted C does what our nest says. Accepted only when the TOTAL number
            # of reduction axes matches on both sides and ours is the single nest,
            # which is what makes it a staging difference rather than a different
            # amount of accumulation.
            mine_red = sum(o["iter"].count("reduction") for o in m["ops"])
            real_red = sum(o["iter"].count("reduction") for o in r["ops"])
            single = len([o for o in m["ops"] if "reduction" in o["iter"]]) == 1
            chained = len([o for o in r["ops"] if "reduction" in o["iter"]]) > 1
            # A MULTI-PASS REDUCTION, where the compiler emits ONE NEST PER PASS.
            #
            # Different from the staging case above and it needs its own evidence.
            # `var_mean` is TWO passes over the same axis of the same operand by
            # definition -- the variance is the mean of squared deviations from
            # the mean, so the mean must exist before pass two starts -- and
            # torch-mlir materialises each pass as its own `P,R` generic. Measured:
            # `fp32_std_rows` is 1 nest here and 2 there, `fp32_var_mean_rows` 1
            # and 3. Total reduction AXES therefore differ, which is why the
            # staging rule above correctly refuses it.
            #
            # Accepted only with attribution: the target must be in
            # `MULTI_PASS_REDUCTIONS`, which is derived from the emitter that
            # actually writes two traversals. Without that, "the compiler used
            # more nests" excuses any nest at all.
            if not (single and chained and mine_red == real_red):
                mine_ops = [o for o in other_ops if "reduction" in o["iter"]]
                tgts = {o.get("target") for o in mine_ops}
                same_shape = len({(sig, maps) for sig, maps in red_g}) == 1
                if (single and chained and same_shape
                        and tgts and tgts <= _MULTI_PASS):
                    multipass = [f"ours 1 nest, linalg {len(red_g)} identical "
                                 f"{red_g[0][0]} nests -- one per pass of "
                                 f"{sorted(tgts)}"]
                    unmatched, pool = [], []
            if single and chained and mine_red == real_red:
                staged = [f"ours 1 nest of {mine_red} reduction axes, "
                          f"linalg {len([o for o in r['ops'] if 'reduction' in o['iter']])} chained"]
                unmatched, pool = [], []
            elif multipass:
                pass
            else:
                # A FUSED BROADCAST OPERAND. Ours carries one map the compiler's
                # nest does not, and it belongs to an extra INPUT rather than to a
                # different access pattern: `addmm`'s bias, rank 1 and stride 0
                # down the rows, which we fold into the accumulator's initial
                # value while torch-mlir materialises it as a separate broadcast
                # outside the matmul.
                #
                # Accepted only when the surplus is provably a broadcast of an
                # operand this nest already indexes: same signature, EVERY one of
                # the compiler's maps present in ours, and each surplus map's loop
                # set a strict SUBSET of the union of the matched maps' loops --
                # so it introduces no new loop, no new stride, and no new extent.
                # A surplus map naming a loop the compiler's nest does not use is
                # still a DIFFER.
                if len(red_m) == 1 and len(red_g) == 1:
                    (sig_m, maps_m), (sig_g, maps_g) = red_m[0], red_g[0]
                    surplus = list(maps_m)
                    for mp in maps_g:
                        if mp in surplus:
                            surplus.remove(mp)
                    loops_g = set()
                    for mp in maps_g:
                        loops_g |= _map_loops(mp)
                    if (sig_m == sig_g and len(surplus) == len(maps_m) - len(maps_g)
                            and surplus
                            and all(_map_loops(x) < loops_g for x in surplus)):
                        fused_op = [f"[{sig_m}] ours folds {surplus} in as a "
                                    "broadcast operand; linalg materialises it "
                                    "outside the nest"]
                        unmatched, pool = [], []
                if not fused_op:
                    notes.append(f"reduction nests/maps ours={red_m} linalg={red_g}")
        # A RANK-0 NEST IS NOT A LOOP NEST. `all(x)` decomposes to
        # logical_not -> any.dims -> logical_not, and that last `logical_not` runs
        # on the SCALAR the reduction produced: our table records it with
        # `iterator_types = []`, extents `[]`, and maps `() -> ()`. It has no
        # iteration space, so there is no access pattern for it to disagree about
        # and no loop to tile, vectorise or reorder.
        #
        # Set aside from the surplus comparison rather than counted as a nest,
        # because counting it made `fp32_all` a DIFFER whose note read
        # "all-parallel nests not in linalg: ['P,P']" -- naming the WRONG nest and
        # sending the reader after the elementwise pass instead of the scalar tail.
        # Reported in the verdict so it is set aside visibly, never silently.
        #
        # SYMMETRIC, and it has to be: the compiler emits rank-0 generics of its
        # own (the generalised `linalg.fill` that seeds `all`'s accumulator with
        # `true`, which our table has no counterpart for at all). Filtering only
        # our side left `par_g == ['']` and the surplus rule still could not
        # explain the difference -- a one-sided normalisation moves the
        # disagreement rather than resolving it.
        #
        # Narrow on purpose: rank 0 means zero loops AND every map rank 0. An op
        # with loops is compared like any other.
        def _rank0(o):
            return (not o["iter"]
                    and all(not _map_loops(mp) for mp in (o.get("maps") or [""])))

        scalar_ops = [o for o in other_ops if _rank0(o)]
        other_ops = [o for o in other_ops if not _rank0(o)]
        real_ops = [o for o in r["ops"] if not _rank0(o)]
        scalar_g = len(r["ops"]) - len(real_ops)
        par_m = [_sig(o) for o in other_ops if "reduction" not in o["iter"]]
        par_g = [_sig(o) for o in real_ops if "reduction" not in o["iter"]]
        extra = [s for s in par_m if par_m.count(s) > par_g.count(s)]
        # A VALUE-MOVING OP THE COMPILER FOLDED IS NOT A DISAGREEMENT ABOUT ACCESS
        # PATTERN. Measured on `fp16_hardswish`: our table has 7 all-parallel nests
        # and linalg has 5, and the two extras are both `aten._to_copy` -- the
        # fp16->fp32 cast in and the fp32->fp16 cast out. torch-mlir folds those into
        # the arithmetic as `arith.extf`/`truncf` inside the neighbouring generics
        # rather than materialising a loop for each.
        #
        # This check exists to catch a WRONG NEST, and a cast's nest is the same
        # elementwise nest as the op it feeds. So it is accepted as its own verdict
        # -- never silently -- and only when BOTH conditions hold: every surplus nest
        # belongs to an op whose emitted body just moves a value, and its maps are
        # all identity. An extra nest from an op that computes something is still a
        # DIFFER.
        folded, addressed = [], []
        if extra:
            # THE COMPILER FUSES; THIS TABLE IS PER-PRIMITIVE BY DESIGN. Measured:
            # `fp16_gelu` is 7 all-parallel nests here and ONE linalg.generic;
            # `fp32_logaddexp` is 15 and four. That is what fusion means, not a
            # disagreement -- a chain of same-shape elementwise passes can be
            # merged into any number of loops without changing which element reads
            # which, and this module's own docstring says the annotation is a
            # property of the primitive.
            #
            # Accepted only when it really is that situation: every all-parallel
            # nest of ours has the SAME iterator signature and identity maps (so no
            # broadcast or transpose is hiding in the surplus), and linalg has at
            # least one nest with that signature. A surplus nest whose maps differ
            # from its neighbours is still a DIFFER.
            # FUSION NEEDS EVIDENCE, and the evidence is that each surplus nest
            # came from a NAMED PRIMITIVE. Without that, "the compiler fused N real
            # ops" is indistinguishable from "we invented a duplicate nest" -- the
            # two are the same data. So a nest with no `target` cannot establish
            # fusion and still reports DIFFER, which is what keeps
            # `test_an_all_parallel_nest_we_invented_is_still_flagged` meaningful.
            par_ops = [o for o in other_ops if "reduction" not in o["iter"]]
            sigs_m = {_sig(o) for o in par_ops}
            attributed = all(o.get("target") for o in par_ops)
            # A RANK-0 OPERAND IS A CONSTANT, NOT AN ACCESS PATTERN. `ident`
            # required every operand map to be the full loop nest, which rejected a
            # nest reading a SCALAR alongside its tensors -- and 7 of the 15 DIFFERs
            # in batches 36-43 were exactly that: `hardshrink`, `heaviside`,
            # `kl_div`, `logit`, `log_sigmoid`, `lerp.Scalar` and
            # `binary_cross_entropy_with_logits` each decompose to a
            # `scalar_tensor`/`full` feeding a `where` or a compare, giving a surplus
            # operand map of `(d0, d1) -> ()`. The rank-0 PRODUCER was already set
            # aside as "not a loop nest"; its CONSUMER's rank-0 operand was not, so
            # the kernel was reported as disagreeing with the compiler about a
            # constant. `_identity_but_for_scalars` applies the module's existing
            # rank-0 judgement at operand granularity.
            #
            # A `(d0, 0)` BROADCAST IS STILL REFUSED, and the difference is not
            # cosmetic: a size-1 axis is an axis, the operand is a real tensor, and
            # which element an iteration reads is a live question about it. An
            # earlier draft of this rule accepted any projection -- which turned
            # `test_a_surplus_nest_with_a_DIFFERENT_map_is_still_DIFFER` green and is
            # how the two cases were told apart rather than lumped.
            #
            # `len(sigs_m) == 1` is likewise kept. A surplus spanning ranks is not
            # explained by this rule; where it occurs (`tril`'s rank-1 arange feeding
            # a rank-2 mask) the compiler turns out not to have fused a broadcast at
            # all but to have replaced it with `linalg.index`, which is the next rule
            # and carries its own evidence.
            ident = all(_identity_but_for_scalars(o) for o in par_ops)
            if (attributed and len(sigs_m) == 1 and ident
                    and sigs_m <= set(par_g)):
                movers = sorted({o["target"] for o in other_ops
                                 if "reduction" not in o["iter"]
                                 and o["target"] in VALUE_MOVING})
                folded = [f"ours {len(par_m)} per-primitive nests, linalg "
                          f"{len(par_g)} fused"]
                if movers:
                    folded.append(f"includes value-moving {movers}")
                extra = []
            elif (attributed and _addressing_note(r.get("addressing"))
                    and not any(_is_transpose(o) for o in par_ops)):
                # THE COMPILER PUT THE ADDRESS ARITHMETIC WHERE THIS COMPARISON
                # CANNOT SEE IT. Three kernels in batches 36-43, and the surplus map
                # is a real access pattern in every one -- so the fusion rule above
                # correctly refuses them and this is not a second chance at it.
                #
                #   `fp32_diag`, `fp32_trace`  ours `(d0) -> (d0, d0)`, the diagonal
                #       gather. The compiler emits ONE identity nest whose body calls
                #       `linalg.index 0` and does a scalar `tensor.extract` at (i, i).
                #   `fp32_glu`                 ours `(d0, d1) -> (d0, d1 + 512)`. The
                #       compiler emits a `tensor.extract_slice` -- a VIEW, not a
                #       generic -- and both its generics are then plain identity.
                #
                # In both the access is identical and the difference is only WHERE it
                # is written: in an indexing map on our side, in the loop body or a
                # view op on the compiler's. A comparison of iterator types and
                # indexing maps is structurally blind to the latter, so the verdict
                # names its own blind spot instead of reporting a disagreement it
                # cannot substantiate.
                #
                # THE EVIDENCE IS THE COMPILER'S OWN IR. `r["addressing"]` is
                # collected in the child, where the full IR still exists -- by the
                # time `L.summary` has kept only the generics, the `extract_slice`
                # that DEFINES a generic's operand is already gone. Without the
                # token this stays a DIFFER, so a table that really does invent a
                # gather is still caught.
                # NAME WHICH OF THE TWO SHAPES THIS IS. Both are index arithmetic the
                # compiler moved out of the maps, but they are not the same thing on
                # our side and a reader needs to know which: `diag`/`trace`/`glu`
                # carry a real offset or gather IN the map, while
                # `tril`/`triu`/`diag_embed`/`safe_softmax` carry only broadcasts of
                # an iota the compiler replaced with `linalg.index`. Printing an empty
                # list for the second case reads as "no evidence" when the evidence is
                # just of a different kind.
                outside_ops = sorted({o["target"] for o in par_ops
                                      if _addressing_outside_maps(o)})
                if outside_ops:
                    ours = f"ours spells the access in an indexing map {outside_ops}"
                else:
                    ours = ("our surplus is index-derived broadcasts from "
                            f"{sorted({o['target'] for o in par_ops})}")
                addressed = [f"{ours}; linalg puts it in "
                             f"{_addressing_note(r.get('addressing'))} -- the body or "
                             f"a view, neither of which this comparison reads"]
                extra = []
            elif attributed and ident and red_g and red_m == red_g:
                # A PRODUCER FUSED INTO ITS REDUCTION CONSUMER, which the rule
                # above cannot reach: it requires linalg to still have a PARALLEL
                # nest of OUR surplus signature to fuse into, and here there is
                # none. Measured on `fp32_all`: `all(x)` decomposes to
                # logical_not -> any.dims -> logical_not, so our table has a `P,P`
                # map feeding an `R,R` reduce, and torch-mlir emits ONE `R,R`
                # generic whose body is `cmpf une` then `andi` -- the elementwise
                # pass moved inside the reduction's loop body. Same computation,
                # one traversal instead of two plus a 256 KB intermediate.
                #
                # THE EVIDENCE IS THAT THE REDUCTIONS AGREE EXACTLY.
                # `red_m == red_g` compares signature AND every normalised map of
                # every reduction nest on both sides. When that holds, the only
                # difference between the two schedules is elementwise passes that
                # the compiler did not materialise -- which is what fusion is.
                # Combined with `ident` (no broadcast or transpose hiding in the
                # surplus) and `attributed` (each surplus nest came from a named
                # primitive), that is a complete account of the difference.
                #
                # An earlier version instead required all surplus nests to share ONE
                # signature and matched their map against the reduction's operands.
                # That worked for `fp32_all` and failed for `fp32_all_dim`, where
                # `all` along a dim leaves TWO surplus nests of different ranks --
                # the rank-2 `logical_not` before the reduce and a rank-1 one after
                # it, both of which torch-mlir absorbs. Requiring one signature
                # reported a correct kernel as a DIFFER, and the map test could not
                # have rescued it: the rank-1 nest's identity map is spelled `(d0)`
                # while the reduction's result map is `(d1)` -- the same access
                # pattern under a different loop name.
                #
                # NO "AND LINALG HAS NO PARALLEL NEST AT ALL" CLAUSE. The first
                # version had `not par_g`, which worked for `fp32_all` -- a full
                # reduction, whose accumulator seed is a RANK-0 fill and therefore
                # already set aside -- and failed for `fp32_all_dim`, where the
                # reduction keeps a dimension so the fill is a rank-1 PARALLEL
                # generic. A rank-1 fill is not a counterpart to a `P,P` producer,
                # and treating its presence as disqualifying reported a correct
                # kernel as a DIFFER. The guard is unnecessary anyway: this is the
                # `elif` of the rule that fires when linalg DOES have a matching
                # parallel nest, so reaching here already means it does not.
                #
                # (Extents would be the stronger check and are not available: the
                # compiler-side summary carries `iter` and `maps` only. The map
                # text is what both sides do share, and an identical map over an
                # identical loop count is the same iteration space unless the two
                # dumps disagree about a dimension the map does not mention.)
                # `red_g`'s maps are stored through `_norm_map`, so ours has to go
                # through it too -- comparing raw text against normalised text
                # silently never matches, and a check that can only fail is worse
                # than no check.
                into = [sig for sig, _maps in red_g]
                if into:
                    into_red = [f"ours {len(par_m)} parallel nest(s) "
                              f"{sorted(sigs_m)} + {len(red_m)} reduction, linalg "
                              f"{len(red_g)} reduction only, and the reduction "
                              f"nests agree EXACTLY ({into[0]}), so the surplus is "
                              f"elementwise passes the compiler did not materialise",
                              f"fused: {sorted({o['target'] for o in par_ops})}"]
                    extra = []
        if extra:
            notes.append(f"all-parallel nests not in linalg: {sorted(set(extra))}")
        if notes:
            rows.append((name, "DIFFER", "; ".join(notes)))
        elif canonical:
            rows.append((name, "canonical",
                         "identical after renaming loops to the result map and "
                         f"dropping our extent-1 reduction: {canonical}"))
        elif staged:
            rows.append((name, "staged-reduce",
                         "same total accumulation, linalg stages it as a chain: "
                         + "; ".join(staged)))
        elif outside:
            rows.append((name, "outside-linalg",
                         "the compiler expressed this op in another dialect, so "
                         "there is no linalg reduction nest to compare: "
                         + ", ".join(outside)))
        elif fused_op:
            rows.append((name, "fused-operand",
                         "same nest and maps, plus a broadcast operand the "
                         "compiler materialises outside it: " + "; ".join(fused_op)))
        elif multipass:
            rows.append((name, "multi-pass",
                         "same axis reduced more than once, one linalg nest per "
                         "pass: " + "; ".join(multipass)))
        elif into_red:
            rows.append((name, "fused-into-reduce",
                         "the compiler folded our elementwise producer into the "
                         "body of its reduction consumer: " + "; ".join(into_red)))
        elif addressed:
            rows.append((name, "addressing-in-body",
                         "same access pattern, written where an indexing-map "
                         "comparison cannot read it: " + "; ".join(addressed)))
        elif folded:
            rows.append((name, "fused", "; ".join(folded)))
        elif extras:
            rows.append((name, "extra-result",
                         "same nest and maps, plus a duplicate result: "
                         + " ".join(extras)))
        else:
            checked = f"{len(red_m)} reduction nest(s) + maps" if red_m \
                else "all-parallel only"
            rows.append((name, "EXACT", checked))
        if scalar_ops or scalar_g:
            nm, vd, note = rows[-1]
            rows[-1] = (nm, vd, note + "; set aside rank-0 (no loops, nothing to "
                        f"schedule): ours {sorted({o.get('target') for o in scalar_ops})}"
                        f", linalg {scalar_g}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True,
                    help="interpreter with torch_mlir installed")
    ap.add_argument("--batch", type=int, action="append", default=None)
    args = ap.parse_args(argv)
    batches = args.batch or sorted(_all_batches())

    rows = compare(derived(batches), real_linalg(args.python, batches))
    width = max(len(n) for n, _, _ in rows)
    for name, verdict, note in rows:
        print(f"  {name:{width}s}  {verdict:18s} {note}")

    differ = [r for r in rows if r[1] in ("DIFFER", "ERROR")]
    print(f"\n{sum(1 for r in rows if r[1] == 'EXACT')} exact, "
          f"{sum(1 for r in rows if r[1] == 'canonical')} canonical, "
          f"{sum(1 for r in rows if r[1] == 'extra-result')} extra-result, "
          f"{sum(1 for r in rows if r[1] == 'fused')} fused, "
          f"{sum(1 for r in rows if r[1] == 'staged-reduce')} staged-reduce, "
          f"{sum(1 for r in rows if r[1] == 'scan')} scan (no linalg form), "
          f"{sum(1 for r in rows if r[1] == 'multi-pass')} multi-pass, "
          f"{sum(1 for r in rows if r[1] == 'fused-operand')} fused-operand, "
          f"{sum(1 for r in rows if r[1] == 'fused-into-reduce')} fused-into-reduce, "
          f"{sum(1 for r in rows if r[1] == 'addressing-in-body')} "
          f"addressing-in-body, "
          f"{sum(1 for r in rows if r[1] == 'no-linalg')} no-linalg-form, "
          f"{sum(1 for r in rows if r[1] == 'outside-linalg')} outside-linalg, "
          f"{len(differ)} differ/error")
    if differ:
        print("\nA DIFFER means the hand-written schedule table disagrees with "
              "the compiler. Fix the table before trusting kernels built from it.")
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
