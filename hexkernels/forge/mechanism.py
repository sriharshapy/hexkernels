"""Which hardware mechanisms a kernel is entitled to, derived from its size.

THE PROVENANCE ARGUMENT
-----------------------
A benchmark task is three things: an **op**, a **size**, and the **hardware
mechanisms** it should exercise. The first two are harvested from public sources
(the operator registry; the shape ladder). The third is the one that would
otherwise be an author's opinion -- and an opinion is not defensible in a paper.

This module removes the opinion. The mechanism is *computed from the size*
against the probed memory hierarchy, so the justification for "this task should
use DMA" is arithmetic anyone can re-run:

    working set <= L1D            -> T0: registers/L1, no staging to speak of
    L1D < working set <= L2       -> T1: L2-resident; prefetch pays, staging does not
    L2 < working set <= VTCM      -> T2: staging into VTCM is what buys the bandwidth
    working set > VTCM            -> T3: stream it -- DMA double-buffering

That ladder is why a task can *demand* DMA without being unfair: below L2 the
data already fits, so a DMA kernel would not accelerate, and grading against it
would be grading noise. `hexkernels.forge.frontend.oracle` states the same boundary
from the other direction (`dma_justified`).

WHAT "REALLY USES THE INTRINSICS" MEANS
--------------------------------------
The selection here answers a narrower question than "can we emit C++ for it":
does the op, at a size that fits the ladder, *have a reason* to touch HVX, HMX,
DMA, VTCM or l2fetch? An op that is pure metadata (`view`, `expand`), pure
control flow, or has no arithmetic at all can be emitted perfectly and still
teach nothing about the hardware. Those are excluded from the training corpus
rather than emitted and scored.

Mechanism eligibility is a claim about the OPPORTUNITY, not about a particular
kernel. Whether a given candidate actually used the mechanism is a separate,
evidence-based question answered after execution by `hexkernels.anticheat.anticheat` and
`anticheat_runtime` -- static presence, then executed-and-not-futile. Nothing
here is ever evidence that a mechanism was used.
"""
from dataclasses import dataclass

from hexkernels.core import target as _target
from hexkernels.forge.frontend import schedule as _sched

# Mechanism names, matching the anti-cheat's flags so a plan and a verdict can be
# compared field for field.
HVX = "hvx"
HMX = "hmx"
DMA = "dma"
VTCM = "vtcm"
L2FETCH = "l2fetch"

TIERS = ("T0", "T1", "T2", "T3")

# HMX is an int8/fp16 tile engine. Anything wider has no tile multiply to call,
# so the mechanism cannot be granted however well the loop shape matches.
HMX_MAX_DTYPE_BYTES = 2


@dataclass(frozen=True)
class Plan:
    """What this op at this size is entitled to use, and why."""

    tier: str
    working_set_bytes: int
    mechanisms: frozenset
    reasons: tuple          # (mechanism, one-line justification)

    def explain(self) -> str:
        head = (f"tier {self.tier}, working set {self.working_set_bytes} B "
                f"vs L1D/L2/VTCM "
                f"{_target.current().l1d_bytes}/{_target.current().l2_bytes}/"
                f"{_target.current().vtcm_bytes}")
        return "\n".join([head] + [f"  {m}: {why}" for m, why in self.reasons])


def tier_for(working_set: int, tgt=None) -> str:
    """Place a working set on the memory-hierarchy ladder.

    Boundaries are inclusive at the top of each level: a working set of exactly
    L2 still fits in L2, so it is T1 and staging is not justified. That
    convention matches `sizing.dma_justified`, which uses a strict `>`.
    """
    t = tgt or _target.current()
    if working_set <= t.l1d_bytes:
        return "T0"
    if working_set <= t.l2_bytes:
        return "T1"
    if working_set <= t.vtcm_bytes:
        return "T2"
    return "T3"


def _contracts(sched, graph) -> bool:
    """Is this primitive a contraction -- parallel loops over a shared reduction
    accumulating a product? That nest, and only that nest, is what the HMX tile
    matmul implements.

    `aten.mm` always is. `aten.convolution` is too, but ONLY when `groups == 1`:
    a grouped or depthwise convolution reduces over `kh`/`kw` within one channel
    and contracts nothing across channels, so there is no tile matmul to map it
    onto -- granting `hmx` there would create an entitlement no correct kernel
    can satisfy, which then reads as a model failure in the results table.
    """
    if sched.target in _sched.MATMUL_TARGETS:
        return True
    if sched.target in _sched.CONV_TARGETS:
        node = next((n for n in graph.nodes if n.name == sched.result.name), None)
        return bool(node is not None and node.args[8] == 1)
    return False


def plan_for(graph, dtype_bytes=4, tgt=None) -> Plan:
    """The mechanisms a traced graph at its traced size is entitled to use."""
    t = tgt or _target.current()
    ws = _sched.working_set_bytes(graph, dtype_bytes=dtype_bytes)
    tier = tier_for(ws, t)
    scheds = _sched.annotate(graph)

    mechs, why = set(), []

    # --- compute mechanisms: from the SCHEDULE ---------------------------
    if any(s.vectorizable_loop() is not None for s in scheds):
        mechs.add(HVX)
        why.append((HVX, "at least one loop is parallel with unit/zero stride on "
                         "every operand, so it maps to a vector loop"))
    elif any(s.reduction_loops for s in scheds):
        mechs.add(HVX)
        why.append((HVX, "reduction loops vectorise through an accumulator plus a "
                         "horizontal reduce"))
    elif any(s.scan_loops for s in scheds):
        # A SCAN AXIS IS NOT PARALLEL, and the first two clauses correctly refuse
        # it: iteration k reads what k-1 wrote, so there is no vector store to
        # licence. But an ASSOCIATIVE scan is still a vector algorithm -- shift
        # by 1,2,4,...,lanes/2 and combine at each step (Hillis-Steele), which
        # computes the whole prefix in log2(lanes) operations. Withholding HVX
        # here would report the scan family as un-accelerable, which is false.
        mechs.add(HVX)
        why.append((HVX, "the scanned axis carries a dependence, so it is not a "
                         "plain vector loop -- but the combine is associative, so "
                         "it vectorises through log2(lanes) shift-and-combine "
                         "steps instead of through an accumulator"))

    if any(_contracts(s, graph) for s in scheds):
        # SHAPE IS NECESSARY BUT NOT SUFFICIENT. Two parallel loops over a shared
        # reduction is the shape HMX implements -- but HMX is an int8/fp16
        # datapath, and there is no fp32 tile multiply to call. Granting `hmx` on
        # an fp32 matmul creates an entitlement no correct kernel can satisfy,
        # which then reads as a model failure in the results table.
        #
        # Caught on batch 1: matmul_fp32 was entitled to hmx and correctly did
        # not use it. Demoting the operands to fp16 to reach the tile engine
        # spends ~5e-4 relative error on the inputs alone against a 1e-3
        # tolerance on the result of a 256-term reduction -- most of the budget
        # before a single product is formed.
        if dtype_bytes <= HMX_MAX_DTYPE_BYTES:
            mechs.add(HMX)
            why.append((HMX, "parallel loops over a shared reduction of products "
                             "is the shape the HMX tile matmul implements, and the "
                             f"dtype ({dtype_bytes} B) fits its int8/fp16 datapath"))
            # STILL GATED ON *INPUT* WIDTH ONLY -- known over-grant, unchanged.
            # int8 x int8 accumulates to int32, and HMX has no int32 accumulator
            # readout (store types are .ub/.uh/.hf/.f8), so an int8 contraction
            # that must return int32 cannot round-trip through the tile engine
            # however well the nest matches. Recorded as an open decision rather
            # than narrowed here, because the fix is an output-dtype predicate,
            # not a constant edit. Convolution inherits the same caveat.
        else:
            why.append((HMX, f"NOT granted: the shape matches, but a {dtype_bytes}-byte "
                             "dtype exceeds the int8/fp16 HMX datapath"))

    # --- memory mechanisms: from the SIZE --------------------------------
    # This is the half that must not be an opinion. Each line is a comparison
    # against a probed constant, not a judgement about the op.
    if ws > t.l1d_bytes:
        mechs.add(L2FETCH)
        why.append((L2FETCH, f"working set {ws} B exceeds L1D {t.l1d_bytes} B, so "
                             "the stream misses L1 and prefetch has something to hide"))
        # KNOWN OVER-GRANT, measured 2026-08-03. This rule uses CAPACITY only,
        # and capacity is not sufficient for l2fetch the way it is for DMA/VTCM.
        # An l2fetch is warranted when the access pattern is a known
        # width x height x stride region the hardware prefetcher will NOT
        # already cover; a purely sequential stream is exactly the case it does
        # cover. Measured on a 256 KB sequential fp32 relu, identical kernel
        # otherwise: 4836 cycles with no l2fetch, 5276 with one -- the prefetch
        # cost ~9% and bought nothing. So at T1 with a unit-stride walk this
        # entitlement is generous. Left in place rather than silently narrowed
        # because tightening it means adding a pattern predicate (stride and
        # reuse distance from the indexing maps), which is a design change, not
        # a constant edit. See docs/hexagon/SDK_DOC_INDEX.md, l2fetch section.
    if ws > t.l2_bytes:
        mechs.add(DMA)
        mechs.add(VTCM)
        why.append((DMA, f"working set {ws} B exceeds L2 {t.l2_bytes} B, so the data "
                         "does not stay resident and staging is what buys bandwidth"))
        why.append((VTCM, "the staged tiles need a software-managed scratchpad"))
    if ws > t.vtcm_bytes:
        why.append((DMA, f"working set {ws} B also exceeds VTCM {t.vtcm_bytes} B, so "
                         "it must be streamed in tiles -- double-buffered"))

    return Plan(tier=tier, working_set_bytes=ws, mechanisms=frozenset(mechs),
                reasons=tuple(why))


# --- op selection ----------------------------------------------------------
#
# Ops that emit fine and teach nothing about the hardware. Excluded from the
# training corpus by NAME because there is no structural signal to test: a
# `view` has tensor in and tensor out and a perfectly good schedule; it simply
# performs no arithmetic and moves no data.
NO_ARITHMETIC = frozenset({
    "view", "reshape", "expand", "permute", "transpose", "t", "squeeze",
    "unsqueeze", "detach", "alias", "contiguous", "as_strided", "broadcast_to",
    "flatten", "ravel", "unflatten", "movedim", "moveaxis", "swapaxes",
    "swapdims", "narrow", "select", "slice", "unbind", "chunk", "split",
    "clone", "empty", "empty_like", "zeros_like", "ones_like", "new_empty",
    "item", "size", "numel", "dim", "stride", "is_contiguous",
})


def mechanism_eligible(row) -> bool:
    """Would a kernel for this registry row exercise the accelerator at all?

    A row survives when it is a candidate kernel in a namespace we can trace,
    takes and returns a tensor, and does actual arithmetic. This is the filter
    that keeps the corpus about the hardware rather than about PyTorch's API
    surface.
    """
    if row.get("klass") != "kernel":
        return False
    if row.get("namespace") not in ("aten", "prims"):
        return False
    if not (row.get("has_tensor_in") and row.get("has_tensor_out")):
        return False
    op = row.get("op", "")
    if op in NO_ARITHMETIC:
        return False
    # A row whose only tensor argument is a list (Tensor[]) is a multi-tensor
    # apply, not a single kernel with a schedule.
    tensor_args = [a for a in row.get("args", []) if a.get("is_tensor")]
    if tensor_args and all(a.get("is_list") for a in tensor_args):
        return False
    return True


#: Harvest classes that may appear as a PRIMITIVE inside a selected op's graph.
#:
#: This is deliberately a list of three rather than "anything harvested", because the
#: three it leaves out are each a SIGNAL THAT SOMETHING IS WRONG rather than a taste
#: judgement about kernel-worthiness:
#:
#:   kernel      (1,264 rows) arithmetic. The only class any of the 175 existing
#:               kernels contains -- because the merged rule destroyed the rest.
#:   plumbing    (71)  layout/metadata: `permute`, `view`, `clone`, `unsqueeze`,
#:               `slice`. Exactly what a real decomposition is full of.
#:   no_tensor   (730) does not both take and return a tensor: `scalar_tensor`,
#:               `full`, `full_like`. Inside a graph these materialise a constant,
#:               which every non-trivial op does.
#:
#: REFUSED, and each refusal is diagnostic:
#:   gradient    (256) a backward op. In a FORWARD trace this means the trace is
#:               wrong, not that the kernel needs a derivative.
#:   variant     (1,104) an in-place or `.out` variant. Torch functionalises before
#:               we see the graph, so one appearing means functionalisation did not
#:               run -- and an in-place op has no value semantics to emit.
#:   unsupported_domain (969) sparse, quantised, complex. Outside the pipeline's
#:               dtype scope; `coverage` already has an `out_of_scope` bucket for
#:               these, and admitting them here would only move the failure later.
PRIMITIVE_KLASSES = frozenset({"kernel", "plumbing", "no_tensor"})


def primitive_admissible(row) -> bool:
    """May this op appear INSIDE a kernel's traced graph? A different question from
    `mechanism_eligible`, and conflating the two cost the corpus 66 ops.

    `mechanism_eligible` answers "is this op worth building a kernel FOR" -- a
    SELECTION question, and it is right to say no to `permute`: a standalone
    transpose kernel exercises nothing. But it was also being used to decide whether
    a primitive may appear inside the decomposition of an op that WAS selected, and
    those are not the same question:

      * Is `permute` a good task on its own?              No.
      * Must `scaled_dot_product_attention` transpose K?  Yes, and on this hardware
                                                          moving that data is most
                                                          of the work.

    Attention decomposes to 17 primitives, 2 of which are `permute` and `full_like`.
    Under the merged rule those 2 destroyed all 17. Measured across the whole
    expressible set, 66 ops died that way -- to `scalar_tensor` (22), `clone` (13),
    `permute` (12), `unsqueeze`, `full`, `view`, `slice`, `full_like`. Every one is a
    shape, metadata or constant-materialising op; every one is harvested from
    PyTorch; none is the subject of the kernel that decomposes through it.

    THE PROVENANCE CLAIM IS UNCHANGED AND STILL ENFORCED. It is that the kernel is a
    PyTorch op and every step inside it is a PyTorch op -- so this still requires a
    harvested row in a namespace we trace. What it drops is the extra demand that
    each step ALSO be independently kernel-worthy, which was never a statement about
    where the code came from.

    The two safeguards that keep this from filling the corpus with plumbing live
    elsewhere and are deliberate: a kernel must still be SELECTED on a
    `mechanism_eligible` op (`mine`), and `REPORT.md` counts how many of each
    kernel's primitives are non-arithmetic so the ratio is visible rather than
    inferred.
    """
    if row.get("namespace") not in ("aten", "prims"):
        return False
    return row.get("klass") in PRIMITIVE_KLASSES


def select(rows) -> list:
    """The harvested rows worth building kernels from, in registry order."""
    return [r for r in rows if mechanism_eligible(r)]
