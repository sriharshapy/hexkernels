"""The loop-schedule annotation: iterator types and indexing maps per primitive.

WHY THIS MODULE EXISTS
----------------------
The natural way to get this information is Linalg, MLIR's structured-op dialect,
where every op carries exactly two pieces of schedule metadata::

    linalg.generic {
        iterator_types = ["parallel", "reduction"],
        indexing_maps  = [affine_map<(d0, d1) -> (d0, d1)>,   // operand
                          affine_map<(d0, d1) -> (d0, 0)>]    // result, rank 2
    }

RELATION TO REAL LINALG
-----------------------
torch-mlir IS installed here and `hexkernels.forge.linalg` produces the compiler's
own Linalg for the same graph; `forge2.validate_schedule` diffs this module
against it -- iterator types AND indexing maps, with named ops generalised first
so that `linalg.matmul` and `linalg.conv_2d_nchw_fchw`, whose schedules live in a
C++ op definition rather than the printed IR, are actually compared. Over batches
1-3, all 15 kernels: **10 exact, 1 canonical, 4 extra-result, 0 disagreements.**

This module is kept alongside it for two reasons, not as a stand-in:

  * It answers `vectorizable_loop` -- the innermost parallel loop where *every*
    operand is unit- or zero-stride. That is a Hexagon HVX question about
    contiguous vector loads, not a dialect one, and Linalg does not express it.
  * It has no dependency, so tracing and prompting still work if torch-mlir is
    absent or a nightly regresses.

One decomposition difference worth knowing when comparing maps: torch-mlir routes
softmax's max through `aten.max.dim`, whose result is rank 1 and then `expand`ed,
while PyTorch's own decomposition gives `aten.amax` and keeps rank via
`keepdim=True`. (Its `sum` keeps rank, same as ours -- the difference is specific
to that one op.) Both obey the same rank-preserving map rule -- one result per
operand axis -- so a rank-1 result is `(d0)` and a rank-2 `Nx1` result is
`(d0, 0)`. Comparing the two requires matching on the tensor's rank, not the text.

The information is a *property of the primitive*: `aten.sum.dim_IntList` reduces
over the dims named in its own arguments, an elementwise op is parallel in every
dimension, and `aten.mm` is two parallel loops over a shared reduction. So this
module derives the same two facts directly from the traced `Graph`.

WHY IT IS THE POINT, NOT DECORATION
-----------------------------------
Measured previously on this benchmark: handing a model the scalar reference
closed the *semantics* gap (compute-only tasks went 0 -> 6/12, beating
hand-written prompts) and did nothing at all for memory-schedule tasks (0/7). A
scalar reference carries the arithmetic and no schedule -- it says what to
compute and nothing about how to walk memory. That residual is what this
annotation targets, and it is also what decides which hardware mechanism a kernel
can legitimately use:

    innermost parallel loop, unit stride   -> HVX vector loop
    reduction loop                         -> accumulator + horizontal reduce
    two parallel + one reduction, 2-D      -> HMX matmul shape
    working set vs the memory hierarchy    -> DMA / VTCM staging (see `sizing`)

The last line is the one that matters for task construction: the mechanism is
*derived from the size*, so it needs no separate justification.

REPRESENTATION
--------------
An operand's map is one entry per operand **axis**, each a tuple of `Term`
(`loop`, `coeff`) summed to give that axis's index expression. An axis with no
terms is a constant -- rendered `0`, element stride 0 -- which is exactly a
broadcast or a reduced-away axis.

Most primitives need only "loop k walks axis a with coefficient 1", so `Operand`
also accepts the shorthand `axes=(0, 1, None, ...)` -- one entry per LOOP, giving
the axis it walks -- and normalises it into the general form at construction, so
every consumer reads one representation.

    softmax's sum, out (128,1) from in (128,128):
        loops    = [d0 parallel 128, d1 reduction 128]
        in  (128,128)  axes=(0, 1)     -> maps ((d0,), (d1,))
        out (128,1)    axes=(0, None)  -> maps ((d0,), ())   i.e. (d0, 0)

The general form exists because convolution needs it: its input height axis is
`oh*stride + kh*dilation`, two loops with coefficients indexing one axis, which
the shorthand cannot express. That is also what makes `vectorizable_loop` correct
for a strided convolution -- at stride 2 the input advances two elements per `ow`
step, so the loop is a gather and must be rejected.
"""
from dataclasses import dataclass

from hexkernels.forge.frontend.graph import numel, strides


class UnschedulablePrimitive(NotImplementedError):
    """No schedule rule for this primitive.

    Raised rather than defaulted. A wrong schedule is worse than an absent one:
    it would licence a vectorisation or a mechanism the op cannot actually
    support, and the resulting kernel compiles and produces plausible numbers.
    """


PARALLEL = "parallel"
REDUCTION = "reduction"
# A THIRD ITERATOR KIND, and the reason it has to exist.
#
# Linalg has exactly two, and this module was built to mirror that. A scan
# (`cumsum`, `cumprod`, `cummax`) fits neither: iteration k reads the state that
# iteration k-1 wrote, so the iterations are NOT independent -- reorder them and
# the answer changes -- and the axis is not reduced away either, since the output
# keeps its full extent. Calling it `parallel` would licence a vector store
# across a dependence; calling it `reduction` would licence dropping the axis.
# Both are wrong in the direction that still compiles.
#
# Linalg's own answer is that `linalg.generic` cannot express a scan at all --
# torch-mlir routes cumsum to `tm_tensor.scan`, a different dialect. So this is
# not this module inventing a concept: it is the point at which the two-kind
# vocabulary provably runs out, which is why `validate_schedule` has a verdict
# for it rather than a disagreement.
SCAN = "scan"


@dataclass(frozen=True)
class Loop:
    """One induction variable of the nest."""

    var: str        # d0, d1, ...
    extent: int
    kind: str       # PARALLEL | REDUCTION | SCAN


@dataclass(frozen=True)
class Term:
    """One `d_k * coeff` summand of an operand axis's index expression.

    Exists because convolution's input axis is indexed by TWO loops with
    coefficients -- `ih = oh*stride + kh*dilation` -- which the older
    "loop k walks axis a" table could not express at all. Confirmed against
    `linalg-generalize-named-ops` output for `linalg.conv_2d_nchw_fchw`, which
    at stride 2 / dilation 2 prints
    `(d0..d6) -> (d0, d4, d2 * 2 + d5 * 2, d3 * 2 + d6 * 2)`.
    """

    loop: int
    coeff: int = 1

    def render(self, loops) -> str:
        v = loops[self.loop].var
        return v if self.coeff == 1 else f"{v} * {self.coeff}"


@dataclass(frozen=True)
class Operand:
    """One tensor the op reads or writes, and how the nest indexes it.

    Two ways to state it, one underlying model:

      * `axes[k]` -- the operand axis loop k walks, or None if loop k does not
        index this operand. The common case: one loop per axis, coefficient 1.
      * `maps[a]` -- a tuple of `Term` per operand AXIS, the general affine form.
        For primitives (convolution) whose axes are strided sums of loops.

    Exactly one is given; `axes` is normalised into `maps` at construction so
    every consumer reads a single representation.
    """

    name: str
    shape: tuple
    axes: tuple = None     # one entry per loop: int axis, or None
    is_result: bool = False
    maps: tuple = None     # one entry per operand AXIS: tuple of Term
    nloops: int = None     # loop count; required when `maps` is given directly
    # A CONSTANT ADDEND per operand axis, defaulting to zero everywhere.
    #
    # Added for the shape ops. `slice(dim, start=32, step=2)` indexes its input as
    # `d0 * 2 + 32` and `flip` as `(extent - 1) - d0`: an affine expression with a
    # constant term, which the Term-only model could not express at all -- it could
    # say `d0 * 2` and had nowhere to put the 32. The absence was invisible while
    # every supported op happened to have offset zero, and it would have been
    # invisible in the WRONG direction: a slice's map would have read as if it
    # started at element 0.
    #
    # Strides are unaffected (a constant addend does not change how far a loop step
    # moves), so `stride_for` and `vectorizable_loop` need no change -- which is
    # also why a constant is the right thing to keep separate from the terms.
    offsets: tuple = None

    def __post_init__(self):
        if (self.axes is None) == (self.maps is None):
            raise ValueError(f"{self.name}: give exactly one of `axes` or `maps`")
        if self.maps is None:
            per_axis = [[] for _ in range(len(self.shape))]
            for k, ax in enumerate(self.axes):
                if ax is not None:
                    per_axis[ax].append(Term(loop=k))
            object.__setattr__(self, "maps", tuple(tuple(t) for t in per_axis))
            object.__setattr__(self, "nloops", len(self.axes))
        elif self.nloops is None:
            raise ValueError(f"{self.name}: `maps` requires `nloops`")
        if len(self.maps) != len(self.shape):
            raise ValueError(
                f"{self.name}: {len(self.maps)} map entries for a rank-"
                f"{len(self.shape)} operand -- the map is rank-preserving")
        if self.offsets is None:
            object.__setattr__(self, "offsets", tuple([0] * len(self.shape)))
        elif len(self.offsets) != len(self.shape):
            raise ValueError(f"{self.name}: {len(self.offsets)} offsets for a "
                             f"rank-{len(self.shape)} operand")

    def stride_for(self, loop_index: int) -> int:
        """Element stride of this operand along one loop.

        The general form: sum over axes of (this loop's coefficient in that
        axis's index expression) x (that axis's element stride). 0 when the loop
        appears in no axis -- exactly a broadcast (the same element every
        iteration) or a reduction accumulating into one destination.

        Summing rather than looking up is what makes a strided convolution come
        out right: at stride 2 the input advances 2 elements per `ow` step, so
        the loop is a gather and `vectorizable_loop` must reject it. The old
        int-axis table could not represent that, let alone get it right.
        """
        st = strides(self.shape)
        return sum(t.coeff * st[a]
                   for a, terms in enumerate(self.maps)
                   for t in terms if t.loop == loop_index)

    def affine_map(self, loops) -> str:
        """The Linalg-syntax map, for reporting and for prompting a model.

        RANK-PRESERVING: one result per operand AXIS, so the map's arity matches
        the operand's rank. An axis that no loop indexes is written as the
        constant `0`, never omitted -- a `keepdim` reduction result of
        `tensor<Nx1xf32>` is `(d0, d1) -> (d0, 0)`, and `(d0)` would be the
        rank-1 (`keepdim=False`) form.

        Pinned against real torch-mlir output; dropping the axis instead is a
        mistake this rendering has already made once.
        """
        dims = ", ".join(l.var for l in loops)
        # Axis-first, because the map is defined on the operand's own axes and
        # loop order need not match axis order -- matmul's B operand (K,N) is
        # indexed (d2, d1), and conv's input axis 2 is indexed by d2 AND d5.
        results = []
        for terms, off in zip(self.maps, self.offsets):
            parts = [t.render(loops) for t in terms]
            if off:
                # negative offsets render as `- k`, which is what Linalg prints and
                # what `flip` produces (extent-1 minus the loop).
                parts.append(str(off) if off > 0 or not parts else None)
            expr = " + ".join(x for x in parts if x)
            if off < 0:
                expr = f"{expr} - {-off}" if expr else str(off)
            results.append(expr if expr else "0")
        return f"affine_map<({dims}) -> ({', '.join(results)})>"


@dataclass(frozen=True)
class Schedule:
    """The full annotation for one primitive."""

    target: str
    loops: tuple
    operands: tuple

    @property
    def iterator_types(self) -> tuple:
        return tuple(l.kind for l in self.loops)

    @property
    def result(self):
        return next(o for o in self.operands if o.is_result)

    @property
    def reduction_loops(self) -> tuple:
        return tuple(i for i, l in enumerate(self.loops) if l.kind == REDUCTION)

    @property
    def parallel_loops(self) -> tuple:
        return tuple(i for i, l in enumerate(self.loops) if l.kind == PARALLEL)

    @property
    def scan_loops(self) -> tuple:
        """Loops with a carried dependence.

        Read by `mechanism.plan_for`: a scan axis is not vectorisable as a plain
        loop, but an ASSOCIATIVE scan still vectorises, through log2(lanes)
        shift-and-combine steps (Hillis-Steele) rather than through an
        accumulator. So the presence of a scan loop entitles HVX and the absence
        of a `vectorizable_loop` does not withdraw it.
        """
        return tuple(i for i, l in enumerate(self.loops) if l.kind == SCAN)

    def vectorizable_loop(self):
        """The innermost PARALLEL loop that can become a plain HVX vector loop,
        or None.

        This is the first fact an HVX port needs. The requirement is unit stride
        on the result AND unit-or-zero stride on *every* operand:

          * the result must be unit-stride because a vector store is contiguous;
          * an operand at unit stride is a vector load;
          * an operand at stride 0 is a broadcast -- splat it once, still fine;
          * any other stride is a gather, which is a different (and far more
            expensive) rewrite, so the loop does not qualify.

        Checking only the result is not enough, and `amax` is the case that
        proves it: its result is (128,1), so the outer parallel loop *is*
        unit-stride on the result, while the same loop strides the input by 128.
        Reporting that as vectorisable would licence a contiguous vector load
        across rows that the layout does not support.

        Reduction loops are excluded by construction -- they vectorise through an
        accumulator and a horizontal reduce, which is a different transformation.
        """
        for k in reversed(range(len(self.loops))):
            if self.loops[k].kind != PARALLEL:
                continue
            if self.result.stride_for(k) != 1:
                continue
            if all(o.stride_for(k) in (0, 1) for o in self.operands):
                return k
        return None

    def to_text(self) -> str:
        """Linalg-shaped rendering. This is what gets handed to a model."""
        lines = [f"{self.target}",
                 f"  iterator_types = [{', '.join(repr(t) for t in self.iterator_types)}]",
                 "  loops:"]
        for l in self.loops:
            lines.append(f"    {l.var}: {l.extent} ({l.kind})")
        lines.append("  indexing_maps:")
        for o in self.operands:
            role = "result" if o.is_result else "operand"
            lines.append(f"    {o.name} {tuple(o.shape)} {role}: "
                         f"{o.affine_map(self.loops)}")
        v = self.vectorizable_loop()
        lines.append(f"  vectorizable_loop: "
                     f"{self.loops[v].var if v is not None else 'none'}")
        return "\n".join(lines)


def _loops(extents, kinds) -> tuple:
    return tuple(Loop(var=f"d{i}", extent=e, kind=k)
                 for i, (e, k) in enumerate(zip(extents, kinds)))


def _broadcast_axes(shape, out_rank) -> tuple:
    """Right-align `shape` against a nest of `out_rank` loops.

    Numpy/PyTorch broadcasting: shapes are right-aligned, and an extent of 1
    broadcasts -- the operand does not advance along that loop, so the loop maps
    to no axis. Getting this wrong reads past the buffer and still compiles,
    which is why it is factored out and tested directly.
    """
    off = out_rank - len(shape)
    return tuple(None if k < off or shape[k - off] == 1 else k - off
                 for k in range(out_rank))


def _elementwise(node, shapes) -> Schedule:
    out = node.shape
    kinds = [PARALLEL] * len(out)
    operands = [Operand(name=i, shape=shapes[i],
                        axes=_broadcast_axes(shapes[i], len(out)))
                for i in node.inputs]
    operands.append(Operand(name=node.name, shape=out,
                            axes=tuple(range(len(out))), is_result=True))
    return Schedule(target=node.target, loops=_loops(out, kinds),
                    operands=tuple(operands))


def _reduction(node, shapes) -> Schedule:
    """amax / sum.dim_IntList / mean.dim.

    The nest is over the INPUT's rank, because the reduced axes still have to be
    walked. `node.args[1]` names the reduced dims and `node.args[2]` is keepdim,
    which the traced graphs here always set True -- and that is what makes the
    following broadcast work, since the result keeps rank and carries extent 1
    on the reduced axes.
    """
    src = node.inputs[0]
    in_shape = shapes[src]
    # Same helper the emitter uses, so the nest and the C body cannot disagree
    # about which axes are reduced. See `primitives.reduced_dims`.
    dims = sorted(_reduced_dims(node.target, node.args, len(in_shape)))
    keepdim = len(node.shape) == len(in_shape)

    kinds = [REDUCTION if k in dims else PARALLEL for k in range(len(in_shape))]
    operand = Operand(name=src, shape=in_shape,
                      axes=tuple(range(len(in_shape))))
    # Result axes: with keepdim the result has the input's rank and the reduced
    # loops map to its extent-1 axes -- which is a stride of 0, i.e. every
    # iteration of the reduction accumulates into the same destination element.
    if keepdim:
        res_axes = tuple(None if k in dims else k for k in range(len(in_shape)))
        res_shape = node.shape
    else:
        kept = [k for k in range(len(in_shape)) if k not in dims]
        res_axes = tuple(None if k in dims else kept.index(k)
                         for k in range(len(in_shape)))
        res_shape = node.shape
    result = Operand(name=node.name, shape=res_shape, axes=res_axes, is_result=True)
    return Schedule(target=node.target, loops=_loops(in_shape, kinds),
                    operands=(operand, result))


def _matmul(node, shapes) -> Schedule:
    """aten.mm: (M,K) x (K,N) -> (M,N).

    The canonical Linalg matmul: `["parallel", "parallel", "reduction"]` with
    maps (d0,d2), (d2,d1), (d0,d1). This exact shape is what an HMX tile matmul
    implements in hardware, so recognising it is what licenses that mechanism.
    """
    a, b = node.inputs[0], node.inputs[1]
    (m, k), (k2, n) = shapes[a], shapes[b]
    if k != k2:
        raise UnschedulablePrimitive(
            f"{node.target}: inner dimensions disagree ({k} vs {k2})")
    loops = _loops((m, n, k), (PARALLEL, PARALLEL, REDUCTION))
    return Schedule(
        target=node.target, loops=loops,
        operands=(Operand(name=a, shape=(m, k), axes=(0, None, 1)),
                  Operand(name=b, shape=(k, n), axes=(None, 1, 0)),
                  Operand(name=node.name, shape=(m, n), axes=(0, 1, None),
                          is_result=True)))


def _batch_matmul(node, shapes) -> Schedule:
    """aten.bmm: (B,M,K) x (B,K,N) -> (B,M,N).

    Linalg's `linalg.batch_matmul` generalises to exactly this:
    `["parallel", "parallel", "parallel", "reduction"]` with maps
    (d0,d1,d3), (d0,d3,d2), (d0,d1,d2). The batch loop indexes BOTH operands and
    the result, which is what distinguishes it from a broadcast batch -- and each
    slice is a contraction, so the HMX grant applies per slice.
    """
    a, b = node.inputs[0], node.inputs[1]
    (bsz, m, k), (bsz2, k2, n) = shapes[a], shapes[b]
    if k != k2 or bsz != bsz2:
        raise UnschedulablePrimitive(
            f"{node.target}: shapes do not contract ({shapes[a]} x {shapes[b]})")
    loops = _loops((bsz, m, n, k), (PARALLEL, PARALLEL, PARALLEL, REDUCTION))
    return Schedule(
        target=node.target, loops=loops,
        operands=(Operand(name=a, shape=(bsz, m, k), axes=(0, 1, None, 2)),
                  Operand(name=b, shape=(bsz, k, n), axes=(0, None, 2, 1)),
                  Operand(name=node.name, shape=(bsz, m, n),
                          axes=(0, 1, 2, None), is_result=True)))


def _addmm(node, shapes) -> Schedule:
    """aten.addmm: bias + (M,K) x (K,N) -- the matmul nest with a third operand.

    The bias carries the information this rule exists for. Rank 1 of extent N,
    it is indexed by the column loop and by NOTHING ELSE -- stride 0 down the
    rows and stride 0 along k. That is what an empty term list on an axis means,
    and it is the schedule saying "splat it once, outside both inner loops"
    rather than "load it per element".
    """
    bias, a, b = node.inputs[0], node.inputs[1], node.inputs[2]
    (m, k), (k2, n) = shapes[a], shapes[b]
    if k != k2:
        raise UnschedulablePrimitive(
            f"{node.target}: inner dimensions disagree ({k} vs {k2})")
    bshape = shapes[bias]
    if bshape == (m, n):
        bias_axes = (0, 1, None)
    elif bshape == (1, n):
        bias_axes = (None, 1, None)
    elif bshape == (n,):
        bias_axes = (None, 0, None)
    else:
        raise UnschedulablePrimitive(
            f"{node.target}: bias shape {bshape} is neither a row broadcast nor "
            f"the full ({m}, {n}) result shape")
    loops = _loops((m, n, k), (PARALLEL, PARALLEL, REDUCTION))
    return Schedule(
        target=node.target, loops=loops,
        operands=(Operand(name=bias, shape=bshape, axes=bias_axes),
                  Operand(name=a, shape=(m, k), axes=(0, None, 1)),
                  Operand(name=b, shape=(k, n), axes=(None, 1, 0)),
                  Operand(name=node.name, shape=(m, n), axes=(0, 1, None),
                          is_result=True)))


def _convolution(node, shapes) -> Schedule:
    """aten.convolution: (N,Cin,H,W) x (Cout,Cin/g,KH,KW) -> (N,Cout,OH,OW).

    Nest order and iterator types are Linalg's own, verified against
    `linalg-generalize-named-ops` on `linalg.conv_2d_nchw_fchw`::

        d0=n  d1=oc  d2=oh  d3=ow   parallel
        d4=ic d5=kh  d6=kw          reduction
        input  -> (d0, d4, d2*sh + d5*dh, d3*sw + d6*dw)
        weight -> (d1, d4, d5, d6)
        output -> (d0, d1, d2, d3)

    THREE THINGS THIS RULE IS DELIBERATELY HONEST ABOUT
    ---------------------------------------------------
    * **Padding is not in the map.** Linalg lowers it as a separate `tensor.pad`
      that grows the input; `primitives._emit_convolution` instead keeps the
      original buffer and guards with a bounds `if`. Same arithmetic, and neither
      puts a constant term in the map -- so the maps still compare, but the
      *input extents* differ between the two forms and a comparison must not
      read that as a disagreement.

    * **Grouping is affine only at the two ends.** The input channel is
      `(oc / (Cout/groups)) * Cin_g + ic`. With `groups == 1` the quotient is 0 and
      the index is just `ic`. With `groups == Cout` (depthwise, and every grouped
      case where one filter produces one output channel) the quotient is `oc`
      exactly, so the index is `oc*Cin_g + ic` -- still affine. In between it is a
      genuine floordiv, which this representation cannot express, and this rule
      RAISES rather than emit a map that omits the term.

      Omitting it is not a harmless approximation: a missing `d1` term reads as
      element stride 0 on the input, i.e. "advancing the output channel reads the
      same input plane", which for a depthwise convolution is precisely backwards.
      A schedule that claims a broadcast where the data advances would licence a
      splat, and the kernel would compile and produce plausible numbers.

    * **The `ow` stride is the whole point.** At stride 1 the input advances one
      element per `ow` step and `d3` vectorises; at stride 2 it advances two and
      it must not. That distinction only exists because `Term` carries a
      coefficient.
    """
    inp, wgt = node.inputs[0], node.inputs[1]
    in_shape, w_shape = shapes[inp], shapes[wgt]
    if len(in_shape) != 4 or len(w_shape) != 4:
        raise UnschedulablePrimitive(
            f"{node.target}: only 2-D NCHW convolution is scheduled here, got "
            f"input {in_shape} weight {w_shape}")
    stride, _padding, dilation = node.args[3], node.args[4], node.args[5]
    if node.args[6]:
        raise UnschedulablePrimitive(
            f"{node.target}: transposed=True has a different nest entirely")
    n_batch, _cin, _h, _w = in_shape
    cout, cin_g, kh, kw = w_shape
    _, _, oh, ow = node.shape
    sh, sw = stride
    dh, dw = dilation

    groups = node.args[8]
    if groups == 1:
        chan = (Term(4),)                         # ic alone
    elif groups == cout:
        chan = (Term(1, cin_g), Term(4))          # oc*Cin_g + ic
    else:
        raise UnschedulablePrimitive(
            f"{node.target}: groups={groups} with Cout={cout} makes the input "
            "channel index (oc / (Cout/groups))*Cin_g + ic, a floordiv of a loop "
            "and not an affine expression. Only groups=1 and groups=Cout "
            "(depthwise) are exact here; emitting the map without the oc term "
            "would claim the input does not advance along oc, which is a "
            "broadcast that does not exist.")

    loops = _loops((n_batch, cout, oh, ow, cin_g, kh, kw),
                   (PARALLEL, PARALLEL, PARALLEL, PARALLEL,
                    REDUCTION, REDUCTION, REDUCTION))
    nl = len(loops)
    operands = [
        Operand(name=inp, shape=in_shape, nloops=nl, maps=(
            (Term(0),),                          # n
            chan,                                 # in-channel
            (Term(2, sh), Term(5, dh)),           # ih = oh*sh + kh*dh
            (Term(3, sw), Term(6, dw)),           # iw = ow*sw + kw*dw
        )),
        Operand(name=wgt, shape=w_shape, nloops=nl, maps=(
            (Term(1),), (Term(4),), (Term(5),), (Term(6),))),
        Operand(name=node.name, shape=node.shape, nloops=nl, is_result=True,
                maps=((Term(0),), (Term(1),), (Term(2),), (Term(3),))),
    ]
    bias = node.args[2]
    if isinstance(bias, str):
        # (Cout,), read once per output channel: indexed by d1 alone, so it is
        # stride 0 along every other loop -- a broadcast, which is why it never
        # blocks vectorisation of d3.
        operands.insert(2, Operand(name=bias, shape=shapes[bias], nloops=nl,
                                   maps=((Term(1),),)))
    return Schedule(target=node.target, loops=loops, operands=tuple(operands))


# Primitives whose schedule is "every loop parallel, operands broadcast".
# Membership is what makes an op elementwise; there is no structural test for it.
# DERIVED FROM THE EMITTER TABLES, not written out again.
#
# These were two hand-maintained lists, and they drifted -- which is a defect the
# pipeline cannot survive quietly, because the two stages fail in opposite
# directions. `aten.gt.Tensor` had an emitter and no schedule rule, so batch 4's
# bool kernel emitted C perfectly and then died at `annotate` with tier `-`; the
# reverse (a rule with no emitter) would report a schedule for code that cannot be
# written. Neither list is the real question. The real question is "can the
# pipeline express this primitive", and that has one answer.
#
# The elementwise SCHEDULE is genuinely uniform over the whole family: every loop
# parallel over the output shape, operands broadcast-indexed, which is what
# `_elementwise` computes from the node alone without consulting the target. So
# any op the emitter table treats elementwise has the same nest, and any new row
# added there is scheduled correctly the moment it is added.
from hexkernels.forge.frontend.primitives import (  # noqa: E402  (see above)
    ELEMENTWISE as _EMIT_ELEMENTWISE, REDUCTIONS as _EMIT_REDUCTIONS,
    reduced_dims as _reduced_dims)

from hexkernels.forge.frontend.primitives import (  # noqa: E402
    FILL_CONSTANT as _FILL_CONST, FILL_VALUE_ARG as _FILL_ARG,
    FLAT_COPY as _FLAT_COPY)

ELEMENTWISE_TARGETS = frozenset(_EMIT_ELEMENTWISE) | frozenset({
    # dedicated emitters, same all-parallel nest: optional bounds / a dtype-
    # dependent operator do not change the loop structure.
    #
    # `clamp.Tensor` is here rather than in the table because either of its bounds
    # may be ABSENT and the table would render the missing one as Python `None`
    # straight into the C. Moving it to its own emitter dropped it out of this set
    # -- which is derived FROM the table -- and cost batch 11 its schedule
    # (`UnschedulablePrimitive: no schedule rule for aten.clamp.Tensor`). The
    # coupling is the point of the derivation, so the fix is to name it here, next
    # to the two ops that were already in the same position.
    "aten.clamp.default", "aten.clamp.Tensor", "aten.bitwise_not.default",
}) | frozenset(_FLAT_COPY)

from hexkernels.forge.frontend.primitives import INDEX_REMAP as _REMAP  # noqa: E402

# permute/slice/flip/expand read their operand somewhere other than the output's
# own index, so they are NOT elementwise -- they get their own rule.
REMAP_TARGETS = frozenset(_REMAP)
# arange and the fills have no input operand at all.
GENERATED_TARGETS = frozenset(_FILL_CONST) | frozenset(_FILL_ARG) | frozenset({
    "aten.arange.start_step", "aten.arange.default"})
# A fill has no operand and a flat copy has one whose map is a linearisation
# rather than an affine function of the output loops -- but both are all-parallel
# over the output extent, which is what `_elementwise` computes and what every
# consumer of this set uses it for.

# Same derivation, same reason. `_reduction` reads the reduced dims off the node's
# arguments, so a new row in the emitter's reduction table is scheduled without a
# second edit here.
REDUCTION_TARGETS = frozenset(_EMIT_REDUCTIONS)

MATMUL_TARGETS = frozenset({"aten.mm.default", "aten.bmm.default",
                            "aten.addmm.default"})

# `_convolution` shares `convolution`'s schedule for the same reason it shares its
# emitter: the two schemas agree on their first nine arguments and the extra four are
# backend hints with no effect on the loop nest. One rule, so the two cannot drift.
CONV_TARGETS = frozenset({"aten.convolution.default",
                          "aten._convolution.default"})

# Primitives that contract: two-or-more parallel loops over a shared reduction,
# which is the shape an HMX tile matmul implements. Kept separate from
# MATMUL_TARGETS because that set means specifically "the 2-D (M,K)x(K,N) nest"
# and several consumers rely on that narrower meaning.
CONTRACTION_TARGETS = MATMUL_TARGETS | CONV_TARGETS



def _index_remap(node, shapes) -> Schedule:
    """permute / slice / flip / expand: all-parallel over the OUTPUT, with the
    input's map computed per axis.

    The nest is over the output because that is what the emitted loop walks. The
    input's map is where the op's identity lives, and it is affine in every case:
    a permuted axis is a renamed loop, a slice is `d*step + start`, a flip is
    `extent-1 - d`, an expanded axis is the constant 0.
    """
    from hexkernels.forge.frontend.primitives import INDEX_REMAP
    src = node.inputs[0]
    in_shape = shapes[src]
    out_shape = node.shape
    n = len(out_shape)
    kinds = [PARALLEL] * n
    loops = _loops(out_shape, kinds)

    target = node.target
    maps, offsets = [], []
    if target == "aten.permute.default":
        dims = [d % len(in_shape) for d in node.args[1]]
        inv = [None] * len(in_shape)
        for k, a in enumerate(dims):
            inv[a] = k
        for a in range(len(in_shape)):
            maps.append((Term(inv[a]),) if inv[a] is not None else ())
            offsets.append(0)
    elif target == "aten.slice.Tensor":
        dim = node.args[1] % len(in_shape)
        start = node.args[2] if len(node.args) > 2 and node.args[2] is not None else 0
        step = node.args[4] if len(node.args) > 4 and node.args[4] is not None else 1
        if start < 0:
            start += in_shape[dim]
        start = max(0, min(int(start), in_shape[dim]))
        for a in range(len(in_shape)):
            maps.append((Term(a, step if a == dim else 1),))
            offsets.append(start if a == dim else 0)
    elif target == "aten.flip.default":
        flipped = {d % len(in_shape) for d in node.args[1]}
        for a in range(len(in_shape)):
            if a in flipped:
                maps.append((Term(a, -1),))
                offsets.append(in_shape[a] - 1)
            else:
                maps.append((Term(a),))
                offsets.append(0)
    elif target == "aten.expand.default":
        off = n - len(in_shape)
        for a in range(len(in_shape)):
            broadcast = in_shape[a] == 1 and out_shape[off + a] != 1
            maps.append(() if broadcast else (Term(off + a),))
            offsets.append(0)
    else:
        raise UnschedulablePrimitive(f"no remap rule for {target}")

    operands = (
        Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=n,
                offsets=tuple(offsets)),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True),
    )
    return Schedule(target=target, loops=loops, operands=operands)


def _generated(node, shapes) -> Schedule:
    """arange and the fills: all-parallel over the output, NO input operand.

    A nest with one operand is not a degenerate case to work around -- it is what
    a constant-valued or index-valued output is.
    """
    out_shape = node.shape or ()
    n = len(out_shape)
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(Operand(name=node.name, shape=out_shape,
                                      axes=tuple(range(n)), is_result=True),))



def _diagonal(node, shapes) -> Schedule:
    """One parallel loop indexing BOTH axes of the operand -- the rank-reducing
    case. The map has the same loop in two result positions, which is exactly how
    Linalg writes a diagonal extraction."""
    src = node.inputs[0]
    in_shape = shapes[src]
    n = node.shape[0] if node.shape else 1
    loops = _loops((n,), [PARALLEL])
    return Schedule(target=node.target, loops=loops, operands=(
        Operand(name=src, shape=in_shape, maps=((Term(0),), (Term(0),)), nloops=1),
        Operand(name=node.name, shape=node.shape, axes=(0,), is_result=True)))


def _gather(node, shapes) -> Schedule:
    """index_select: all-parallel over the output, and the operand's map on the
    gathered axis is DATA-DEPENDENT.

    That axis is given no terms -- the honest statement, since the index comes from
    a tensor and no affine expression describes it. A consumer reading `stride_for`
    then sees stride 0 on that axis and will not claim the loop is a unit-stride
    vector walk, which is the conclusion that must not be drawn: a gather is not
    vectorisable as a contiguous load, and on this hardware it must run out of VTCM
    (HVX PRM 3.3).
    """
    src, idx = node.inputs[0], node.inputs[-1]
    in_shape = shapes[src]
    out_shape = node.shape
    n = len(out_shape)
    dim = node.args[1] % len(in_shape)
    maps = [(Term(a),) if a != dim else () for a in range(len(in_shape))]
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=n),
        Operand(name=idx, shape=shapes[idx], maps=((Term(dim),),), nloops=n),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


def _adv_gather(node, shapes) -> Schedule:
    """`aten.index.Tensor` -- advanced indexing: all-parallel over the output, and
    EVERY indexed axis is data-dependent.

    Same statement as `_gather` makes for `index_select`, once per supplied index
    tensor: those axes get NO terms, because the address comes from a tensor and no
    affine expression describes it. A consumer reading `stride_for` then sees stride 0
    there and cannot conclude the loop is a unit-stride vector walk -- which for the
    upsample family is exactly the wrong conclusion, since two of the four axes are
    gathered and the kernel must run out of VTCM (HVX PRM 3.3).

    The axes that were passed `None` keep their affine term, so a `(1, 2, 8, 8)`
    source indexed on its last two axes still reports a contiguous batch and channel
    walk -- which is the part a vectoriser CAN use.
    """
    from hexkernels.forge.frontend.primitives import _adv_index_axes
    src = node.inputs[0]
    in_shape = shapes[src]
    out_shape = node.shape
    n = len(out_shape)
    axes, tens = _adv_index_axes(node.args[1])
    first = axes[0]
    region_rank = n - (len(in_shape) - len(axes))
    maps = []
    for a in range(len(in_shape)):
        if a in axes:
            maps.append(())                      # data-dependent: no affine term
        elif a < first:
            maps.append((Term(a),))
        else:
            maps.append((Term(a + region_rank - len(axes)),))
    operands = [Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=n)]
    for ix in tens:
        # ONE MAP ENTRY PER AXIS OF THE INDEX TENSOR, aligned on the TRAILING axes of
        # the broadcast region -- an `Operand` requires len(maps) == len(shape), and a
        # single entry was accepted only for the rank-1 index tensors. The 1-D
        # upsample ops passed on that accident while every 2-D and 3-D one failed at
        # `annotate` with "1 map entries for a rank-2 operand": `unsqueeze` supplies a
        # (4, 1) index against a (4, 4) region.
        ishape = shapes[ix]
        pad = region_rank - len(ishape)
        imaps = tuple(() if ext == 1 else (Term(first + pad + k),)
                      for k, ext in enumerate(ishape))
        operands.append(Operand(name=ix, shape=ishape, maps=imaps, nloops=n))
    operands.append(Operand(name=node.name, shape=out_shape,
                            axes=tuple(range(n)), is_result=True))
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=tuple(operands))


def _pool(node, shapes) -> Schedule:
    """Pooling: parallel over the output, REDUCTION over the window.

    The input's spatial axis is indexed by TWO loops -- `ih = oh*stride + kh` -- which
    is what `Term`'s coefficient exists for and what convolution already needed. So
    unlike a gather this IS affine and is reported as such: a consumer can see that the
    batch and channel walks are contiguous and that the window is a strided reduction,
    which is the information a vectoriser needs.

    Adaptive pooling's window BOUNDS vary per output, so its reduction extent is the
    worst case (ceil(IN/OUT) + 1) and the emitted C computes the real bounds. The
    extent is an upper bound on the iteration space, which is the honest statement --
    the alternative would be claiming a fixed window the op does not have.
    """
    from hexkernels.forge.frontend.primitives import POOL_MULTI, _pool_params
    src = node.inputs[0]
    in_shape = shapes[src]
    out_shape = node.shape if node.shape is not None else node.results[0][0]
    rank = len(out_shape)
    nsp = rank - 2
    adaptive = "adaptive" in node.target
    if adaptive:
        kern = [-(-in_shape[2 + k] // out_shape[2 + k]) + 1 for k in range(nsp)]
        strd = [1] * nsp
    else:
        kern, strd, _pad = _pool_params(node, nsp)
    loops = _loops(tuple(out_shape) + tuple(kern),
                   [PARALLEL] * rank + [REDUCTION] * nsp)
    maps = [(Term(0),), (Term(1),)]
    for k in range(nsp):
        if adaptive:
            # the window START depends on the output index through a floor division,
            # which is NOT affine -- so the spatial axis gets the window loop only
            maps.append((Term(rank + k),))
        else:
            maps.append((Term(2 + k, strd[k]), Term(rank + k)))
    operands = [Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=rank + nsp)]
    if node.target in POOL_MULTI:
        for i, (rshape, _dt) in enumerate(node.results):
            operands.append(Operand(name=f"{node.name}#{i}", shape=rshape,
                                    axes=tuple(range(rank)), is_result=True))
    else:
        operands.append(Operand(name=node.name, shape=out_shape,
                                axes=tuple(range(rank)), is_result=True))
    return Schedule(target=node.target, loops=loops, operands=tuple(operands))


def _full_gather(node, shapes) -> Schedule:
    """`aten.gather`: all-parallel over the output; the gathered axis is data-dependent
    and the INDEX operand is walked with every loop, not just one.

    That is what separates it from `_gather` (index_select), whose rank-1 index is
    indexed by the gathered loop alone. Here the index has the output's shape, so its
    map is the full identity -- a fact a consumer needs, because it means the index
    read IS contiguous even though the source read is not.
    """
    src, idx = node.inputs[0], node.inputs[-1]
    in_shape = shapes[src]
    out_shape = node.shape
    n = len(out_shape)
    dim = node.args[1] % len(in_shape)
    maps = [(Term(a),) if a != dim else () for a in range(len(in_shape))]
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=n),
        Operand(name=idx, shape=shapes[idx], axes=tuple(range(n))),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


def _pad(node, shapes) -> Schedule:
    """`constant_pad_nd`: all-parallel, unit stride on every axis.

    The pad OFFSET does not appear in the map, and that is correct rather than a
    simplification: `Term` expresses `loop * coeff`, and a constant offset shifts the
    base address without changing the stride or the order. What a consumer needs from
    this rule is that the walk is contiguous -- which it is, and which is why a padded
    copy still vectorises. The bounds test lives in the emitted body, not the schedule.
    """
    src = node.inputs[0]
    out_shape = node.shape
    n = len(out_shape)
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=src, shape=shapes[src], axes=tuple(range(n))),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


def _concat(node, shapes) -> Schedule:
    """`cat`: all-parallel over the output, unit stride, one operand per source.

    THE EMITTED BODY IS N NESTS AND THIS REPORTS ONE, which is a real approximation and
    is stated rather than hidden. The reason it is the right one: every source is copied
    contiguously and the concatenated axis is walked in order, so the two facts a
    consumer takes from a schedule -- is this parallel, and is the walk unit-stride --
    are true of each nest and of the whole. What the single nest cannot express is the
    per-source destination OFFSET, and like `_pad`'s offset that shifts a base address
    without changing stride or order.

    `validate_schedule` will therefore see fewer nests here than the compiler does if
    torch-mlir materialises cat as per-source inserts; that is a difference to report
    with evidence, not to pre-empt by inventing a multi-nest schedule this module has no
    representation for.
    """
    out_shape = node.shape
    n = len(out_shape)
    ops = [Operand(name=s_, shape=shapes[s_], axes=tuple(range(n)))
           for s_ in node.inputs]
    ops.append(Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                       is_result=True))
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=tuple(ops))


def _pdist(node, shapes) -> Schedule:
    """`_pdist_forward`: one parallel loop over PAIRS, one reduction over columns.

    The emitted body is a triangular (i, j > i) nest, and this reports the flattened
    pair index instead -- `N(N-1)/2` parallel iterations. That is the honest shape: it
    states the true iteration COUNT, where a rectangular (N, N) nest would claim twice
    the work plus a diagonal the op does not compute.

    The source operand gets NO affine terms. Both rows a pair reads are functions of
    the pair index through a triangular-number inverse, which is not an affine map --
    the same statement `_gather` makes for a data-dependent axis, and it stops a
    consumer reading the pair loop as a contiguous vector walk. The column axis IS
    affine and is reported as the reduction it is.
    """
    src = node.inputs[0]
    in_shape = shapes[src]
    n_rows, n_cols = in_shape
    n_pairs = n_rows * (n_rows - 1) // 2
    return Schedule(target=node.target,
                    loops=_loops((n_pairs, n_cols), [PARALLEL, REDUCTION]),
                    operands=(
        Operand(name=src, shape=in_shape, maps=((), (Term(1),)), nloops=2),
        Operand(name=node.name, shape=node.shape, axes=(0, None), is_result=True)))


def _conv_transpose(node, shapes) -> Schedule:
    """Transposed convolution: 4 parallel output axes, 3 reduction axes (Cin, KH, KW).

    The same nest shape as a forward convolution, and the input's spatial map is
    NOT affine -- `ih = (oh + pad - kh*dil) / stride` divides, which `Term` (a sum of
    `loop * coeff`) cannot express. Those axes therefore get no terms, the honest
    statement: at stride > 1 only some output positions read any input at all, so a
    consumer must not treat the spatial walk as a unit-stride stream. At stride 1 the
    division vanishes and the relation IS affine, but the rule does not special-case
    that -- it would then describe one instance rather than the operator.

    The weight's map is exact: (Cin, Cout, KH, KW) with input channels FIRST, which is
    the transpose of a forward convolution's layout.
    """
    src, w = node.inputs[0], node.inputs[1]
    n_b, c_in, _h, _w = shapes[src]
    _ci, c_out, kh, kw = shapes[w]
    _n, _co, h_out, w_out = node.shape
    loops = _loops((n_b, c_out, h_out, w_out, c_in, kh, kw),
                   [PARALLEL]*4 + [REDUCTION]*3)
    return Schedule(target=node.target, loops=loops, operands=(
        Operand(name=src, shape=shapes[src],
                maps=((Term(0),), (Term(4),), (), ()), nloops=7),
        Operand(name=w, shape=shapes[w],
                maps=((Term(4),), (Term(1),), (Term(5),), (Term(6),)), nloops=7),
        Operand(name=node.name, shape=node.shape,
                axes=(0, 1, 2, 3, None, None, None), is_result=True)))


def _conv3d(node, shapes) -> Schedule:
    """3-D forward convolution: 5 parallel output axes, 4 reduction (Cin, KD, KH, KW).

    Fully affine, unlike the transposed case: the input's spatial axis is
    `i*stride + k`, two loops with a coefficient, which is exactly what `Term` exists
    for and what `linalg.conv_*` prints. So this reports the real maps rather than
    setting the spatial axes aside -- a consumer can see that at stride 1 the innermost
    walk is unit-stride and vectorisable, and that at stride > 1 it is not.
    """
    src, w = node.inputs[0], node.inputs[1]
    n_b, c_in, _d, _h, _w = shapes[src]
    c_out, _ci, kd, kh, kw = shapes[w]
    _n, _co, d_out, h_out, w_out = node.shape
    st = node.args[4] if len(node.args) > 4 and node.args[4] else [1, 1, 1]
    st = list(st) * 3 if len(st) == 1 else list(st)
    loops = _loops((n_b, c_out, d_out, h_out, w_out, c_in, kd, kh, kw),
                   [PARALLEL]*5 + [REDUCTION]*4)
    return Schedule(target=node.target, loops=loops, operands=(
        Operand(name=src, shape=shapes[src],
                maps=((Term(0),), (Term(5),),
                      (Term(2, st[0]), Term(6)),
                      (Term(3, st[1]), Term(7)),
                      (Term(4, st[2]), Term(8))), nloops=9),
        Operand(name=w, shape=shapes[w],
                maps=((Term(1),), (Term(5),), (Term(6),), (Term(7),), (Term(8),)),
                nloops=9),
        Operand(name=node.name, shape=node.shape,
                axes=(0, 1, 2, 3, 4, None, None, None, None), is_result=True)))


def _searchsorted(node, shapes) -> Schedule:
    """`searchsorted`: all-parallel over the output; the SEARCHED operand gets no map.

    The search loop is NOT one of this module's iterator kinds. It is not parallel (the
    iterations are sequential), not a reduction (nothing is accumulated and the axis is
    not collapsed) and not a scan (iteration k does not read what k-1 wrote so much as
    HALVE the interval, and the trip count is data-independent only in the worst case).
    Rather than invent a fourth kind for one op, the searched axis is left OUT of the
    nest entirely and the sequence operand is given no terms -- the same honest device
    `_gather` uses for a data-dependent address.

    What a consumer needs is here: the output walk is parallel and unit-stride, and the
    sequence is read at an address no affine map describes, so it must not be treated
    as a stream.
    """
    seq, val = node.inputs[0], node.inputs[1]
    out_shape = node.shape
    n = len(out_shape)
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=seq, shape=shapes[seq],
                maps=tuple(() for _ in shapes[seq]), nloops=n),
        Operand(name=val, shape=shapes[val], axes=tuple(range(n))),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


def _trilinear(node, shapes) -> Schedule:
    """`upsample_trilinear3d`: five parallel axes, and NO affine map for the source.

    The source coordinate is `floor((o + 0.5) * IN / OUT - 0.5)` -- a floor of a scaled
    index, which `Term` (a sum of `loop * coeff`) cannot express, and each output reads
    EIGHT neighbours rather than one. So the input gets no terms on its spatial axes:
    the honest statement, and the one that stops a consumer reading the innermost loop
    as a unit-stride stream. The batch and channel axes are exact and reported.
    """
    src = node.inputs[0]
    out_shape = node.shape
    n = len(out_shape)
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=src, shape=shapes[src],
                maps=((Term(0),), (Term(1),), (), (), ()), nloops=n),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


def _repeat(node, shapes) -> Schedule:
    """`repeat`: all-parallel over the output; the source axis is `i % extent`.

    A modulo is not affine, so those axes get no terms -- but the reason differs from a
    gather's and is worth distinguishing: the address is not DATA-dependent, it is a
    known periodic function of the loop. A consumer can still stream the source (it is
    read in order, repeatedly); what it cannot do is treat output and source indices as
    equal. Reporting no term is the conservative statement `Term` can express.
    """
    src = node.inputs[0]
    out_shape = node.shape
    n = len(out_shape)
    in_shape = shapes[src]
    off = n - len(in_shape)
    maps = [() if ext > 1 else () for ext in in_shape]
    return Schedule(target=node.target, loops=_loops(out_shape, [PARALLEL] * n),
                    operands=(
        Operand(name=src, shape=in_shape, maps=tuple(maps), nloops=n),
        Operand(name=node.name, shape=out_shape, axes=tuple(range(n)),
                is_result=True)))


# Singletons: ops with their own rule and no family.
SINGLETON_TARGETS = frozenset({"aten.diagonal.default", "aten.index_select.default",
                               "aten.index.Tensor", "aten.gather.default",
                               "aten.constant_pad_nd.default",
                               "aten.cat.default",
                               "aten._pdist_forward.default",
                               "aten.slow_conv_transpose2d.default",
                               "aten.slow_conv3d_forward.default",
                               "aten.searchsorted.Tensor",
                               "aten.upsample_trilinear3d.default",
                               "aten.repeat.default"})

from hexkernels.forge.frontend.primitives import POOL_TARGETS  # noqa: E402

from hexkernels.forge.frontend.primitives import (  # noqa: E402
    MULTI_REDUCTIONS as _MULTI, SINGLE_OF_MULTI as _SINGLE_OF_MULTI)

# The single-result siblings (var, argmax, argmin) share the multi-result rule:
# same nest, same reduced axes, one result operand instead of two.
MULTI_REDUCTION_TARGETS = _MULTI | frozenset(_SINGLE_OF_MULTI)

from hexkernels.forge.frontend.primitives import (  # noqa: E402
    SCANS as _SCANS, MULTI_SCANS as _MULTI_SCANS, _scan_dim as _emit_scan_dim)

# Derived from the emitter tables, like ELEMENTWISE_TARGETS and
# REDUCTION_TARGETS: a target that can be emitted and has no schedule rule (or
# the reverse) is a build error rather than a silent gap.
SCAN_TARGETS = frozenset(_SCANS) | _MULTI_SCANS

from hexkernels.forge.frontend.primitives import (  # noqa: E402
    SORT_FAMILY as _SORT_FAMILY)
ORDER_TARGETS = frozenset(_SORT_FAMILY)

# EVERY target this module can schedule, in ONE place. The anti-drift test compares
# this against the emitter table, and `schedule_of` dispatches within it -- so the
# test and the dispatcher cannot disagree about what is supported, which is exactly
# the failure mode that let `aten.gt.Tensor` emit C with no schedule rule.
SCHEDULED_TARGETS = (SINGLETON_TARGETS | MULTI_REDUCTION_TARGETS | SCAN_TARGETS
                     | ORDER_TARGETS
                     | REMAP_TARGETS | GENERATED_TARGETS
                     | ELEMENTWISE_TARGETS | REDUCTION_TARGETS
                     | MATMUL_TARGETS | CONV_TARGETS
                      | POOL_TARGETS)



def _multi_reduction(node, shapes) -> Schedule:
    """A reduction with TWO result operands.

    The nest is the ordinary reduction nest -- what differs is that two tensors are
    written from it, so the schedule lists both. `keepdim` is False for these ops in
    the traced graphs (torch drops the reduced axis), so each result's map omits it.
    """
    src = node.inputs[0]
    in_shape = shapes[src]
    rank = len(in_shape)
    dims = sorted(_reduced_dims(node.target, node.args, rank))
    kinds = [REDUCTION if k in dims else PARALLEL for k in range(rank)]
    loops = _loops(in_shape, kinds)
    kept = [k for k in range(rank) if k not in dims]

    operands = [Operand(name=src, shape=in_shape, axes=tuple(range(rank)))]
    results = node.results if node.results else ((node.shape, node.dtype),)
    single = node.target in _SINGLE_OF_MULTI
    for k, (shape, _dt) in enumerate(results):
        # keepdim=False: one result axis per KEPT loop, in order
        axes = tuple(kept.index(i) if i in kept else None for i in range(rank))
        rname = node.name if single else f"{node.name}#{k}"
        operands.append(Operand(name=rname, shape=shape, axes=axes,
                                is_result=True))
    return Schedule(target=node.target, loops=loops, operands=tuple(operands))


def _scan(node, shapes) -> Schedule:
    """One axis carries a dependence; every other axis is parallel.

    The result keeps the operand's shape exactly -- same rank, same extents,
    identity map on both -- which is what separates this from `_reduction`, whose
    result drops or flattens the scanned axis. So the ONLY thing the schedule
    says differently from an elementwise nest is the iterator kind on one loop,
    and that one word is the whole difference between "store a vector" and "you
    cannot store a vector here".
    """
    src = node.inputs[0]
    in_shape = shapes[src]
    rank = len(in_shape)
    dim = _emit_scan_dim(node, rank)
    kinds = [SCAN if k == dim else PARALLEL for k in range(rank)]
    loops = _loops(in_shape, kinds)
    axes = tuple(range(rank))

    operands = [Operand(name=src, shape=in_shape, axes=axes)]
    if node.results:
        # cummax/cummin: value and index, both the operand's shape.
        for k, (shape, _dt) in enumerate(node.results):
            operands.append(Operand(name=f"{node.name}#{k}", shape=shape,
                                    axes=axes, is_result=True))
    else:
        operands.append(Operand(name=node.name, shape=node.shape, axes=axes,
                                is_result=True))
    return Schedule(target=node.target, loops=loops, operands=tuple(operands))


def _order_statistic(node, shapes) -> Schedule:
    """A comparison network over one axis: every other axis is parallel.

    The sorted axis is marked REDUCTION, and the choice needs justifying because
    two other labels look plausible:

      * not `parallel` -- the output element at sorted position p depends on the
        WHOLE input row, not on input position p, so there is no per-iteration
        independence to vectorise across.
      * not `scan` -- a scan's iteration k reads what k-1 wrote, a sequential
        chain. A comparison network has no such chain: bitonic sort is
        log^2(n) DATA-PARALLEL stages, so the axis is more like a reduction that
        happens to emit n values than like a prefix computation.

    `reduction` is the closest of the three and is what makes `mechanism.plan_for`
    grant HVX through the accumulator clause rather than the vector-store clause,
    which is the correct entitlement: these kernels vectorise, but not as a
    unit-stride loop over the sorted axis.

    `sort` keeps the axis at full extent, `topk` narrows it to k, and
    `median`/`kthvalue` drop it -- so the result maps are read from each result's
    own shape rather than assumed.
    """
    src = node.inputs[0]
    in_shape = shapes[src]
    rank = len(in_shape)
    sel, dim_ix, _k = _SORT_FAMILY[node.target]
    dim = (int(node.args[dim_ix]) % rank
           if len(node.args) > dim_ix else rank - 1)
    kinds = [REDUCTION if k == dim else PARALLEL for k in range(rank)]
    loops = _loops(in_shape, kinds)
    kept = [k for k in range(rank) if k != dim]

    operands = [Operand(name=src, shape=in_shape, axes=tuple(range(rank)))]
    for r, (shape, _dt) in enumerate(node.results):
        if len(shape) == rank:
            # sort / topk: the reduced loop indexes the result's own axis too
            axes = tuple(range(rank))
        else:
            # median / kthvalue: the axis is dropped
            axes = tuple(kept.index(i) if i in kept else None for i in range(rank))
        operands.append(Operand(name=f"{node.name}#{r}", shape=shape, axes=axes,
                                is_result=True))
    return Schedule(target=node.target, loops=loops, operands=tuple(operands))


def schedule_of(node, shapes) -> Schedule:
    """The schedule annotation for one node.

    `shapes` maps every value name in the graph to its shape, since a node
    records only its own output shape and the schedule needs its operands'.
    """
    if node.target in _SORT_FAMILY:
        return _order_statistic(node, shapes)
    if node.target in SCAN_TARGETS:
        return _scan(node, shapes)
    if node.target in MULTI_REDUCTION_TARGETS:
        return _multi_reduction(node, shapes)
    if node.target == "aten.diagonal.default":
        return _diagonal(node, shapes)
    if node.target == "aten.index_select.default":
        return _gather(node, shapes)
    if node.target == "aten.gather.default":
        return _full_gather(node, shapes)
    if node.target == "aten._pdist_forward.default":
        return _pdist(node, shapes)
    if node.target == "aten.searchsorted.Tensor":
        return _searchsorted(node, shapes)
    if node.target == "aten.repeat.default":
        return _repeat(node, shapes)
    if node.target == "aten.upsample_trilinear3d.default":
        return _trilinear(node, shapes)
    if node.target == "aten.slow_conv_transpose2d.default":
        return _conv_transpose(node, shapes)
    if node.target == "aten.slow_conv3d_forward.default":
        return _conv3d(node, shapes)
    if node.target == "aten.constant_pad_nd.default":
        return _pad(node, shapes)
    if node.target == "aten.cat.default":
        return _concat(node, shapes)
    if node.target == "aten.index.Tensor":
        return _adv_gather(node, shapes)
    if node.target in POOL_TARGETS:
        return _pool(node, shapes)
    if node.target in REMAP_TARGETS:
        return _index_remap(node, shapes)
    if node.target in GENERATED_TARGETS:
        return _generated(node, shapes)
    if node.target in ELEMENTWISE_TARGETS:
        return _elementwise(node, shapes)
    if node.target in REDUCTION_TARGETS:
        return _reduction(node, shapes)
    if node.target == "aten.bmm.default":
        return _batch_matmul(node, shapes)
    if node.target == "aten.addmm.default":
        return _addmm(node, shapes)
    if node.target in MATMUL_TARGETS:
        return _matmul(node, shapes)
    if node.target in CONV_TARGETS:
        return _convolution(node, shapes)
    raise UnschedulablePrimitive(
        f"no schedule rule for {node.target}; add one to hexkernels.forge.frontend.schedule."
        "frontend.schedule rather than letting the nest be guessed")


def shape_map(graph) -> dict:
    """name -> shape for every value in the graph, placeholders included."""
    shapes = {i.name: i.shape for i in graph.inputs}
    for n in graph.nodes:
        shapes[n.name] = n.shape
    return shapes


def annotate(graph) -> tuple:
    """Schedule every node of a traced graph, in program order."""
    shapes = shape_map(graph)
    return tuple(schedule_of(n, shapes) for n in graph.nodes)


def working_set_bytes(graph, dtype_bytes=4) -> int:
    """Live bytes the whole graph touches: inputs + outputs.

    Deliberately NOT the sum of every intermediate. This is the figure the memory
    hierarchy is compared against to decide whether staging is justified, and
    intermediates are what staging is meant to keep on-chip.
    """
    total = sum(numel(i.shape) for i in graph.inputs)
    by_name = {n.name: n for n in graph.nodes}
    for o in graph.outputs:
        base = o.split("#")[0]
        node = by_name.get(base)
        if node is None:
            continue
        if "#" in o:
            # A MULTI-RESULT OUTPUT. `node.shape` is None for a tuple-producing op,
            # and `numel(None)` raised `TypeError: 'NoneType' object is not
            # iterable` -- so `var_mean` and `max.dim` had no tier at all, the same
            # failure shape as `linalg_vector_norm`'s dim argument. The k-th
            # result's own shape is in `Node.results`.
            k = int(o.split("#")[1])
            if k < len(node.results):
                total += numel(node.results[k][0])
        else:
            total += numel(node.shape)
    return total * dtype_bytes


def graph_text(graph) -> str:
    """The whole annotation, as handed to a model alongside the scalar C++."""
    return "\n".join(s.to_text() for s in annotate(graph))
