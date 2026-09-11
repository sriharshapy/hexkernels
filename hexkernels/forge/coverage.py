"""How much of the harvest can the pipeline actually express? Measured, per op.

WHY THIS EXISTS
---------------
The corpus's only claim is that every kernel traces back to the PyTorch operator
registry. Provenance has been enforceable since batch 3 (`provenance.py`), but
SELECTION was not: the 15 kernels in batches 1-3 were chosen by hand from the
~15 ops `forge/frontend/primitives.py` happens to have emitters for, out of 1,228
eligible harvested rows. So the harvest proved provenance and did not choose
kernels, and "coverage" was an assertion nobody could check.

This module makes it a measurement. For every eligible harvested row it tries to
BUILD the op -- call it through `torch.ops.aten`, trace it with the same
`trace()` the pipeline uses, and emit scalar C with the same `emit_c()` -- and
records which of three things happened:

  covered      traced and emitted. The op is expressible TODAY; a kernel can be
               built from it without touching the pipeline.
  blocked      traced fine, but some decomposed primitive has no emitter. The
               missing primitive names are recorded, and their histogram is the
               actionable output of this whole module: it ranks candidate emitter
               work by how many harvested ops each one unlocks.
  unprobeable  this module could not even call the op (it needs argument values
               we do not synthesise -- a `int[2]` kernel size, an index tensor,
               a list of tensors). Recorded with the reason, so the gap is
               visible rather than silently counted as "not covered".

The three-way split is the point. A two-way covered/not-covered number cannot
tell you whether the pipeline is missing an EMITTER or this prober is missing an
ARGUMENT, and those call for completely different work.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not verify anything. `covered` means C was emitted, not that the C is
correct -- that requires the simulator and the golden harness, which is
`run_batch`'s job and the only thing that ever counts as "working". Coverage is
a map of where kernels COULD come from, not a claim that any of them work.

    python -m hexkernels.forge.coverage                 # the report
    python -m hexkernels.forge.coverage --missing 25    # ranked emitter targets
    python -m hexkernels.forge.coverage --show-covered  # ops a batch can draw on
"""
import argparse
import collections
import json
import re
import sys

import torch

from hexkernels.forge.frontend.emit import emit_c
from hexkernels.forge.frontend.primitives import UnsupportedPrimitive
from hexkernels.forge.frontend.schedule import (UnschedulablePrimitive, annotate,
                                              schedule_of)
from hexkernels.forge.frontend.trace import trace
from hexkernels.forge.mechanism import mechanism_eligible
from hexkernels.forge.provenance import DEFAULT_HARVEST, load_harvest

# Small enough that 1,200 exports are a couple of minutes, and rank-2 so a
# reduction or a matmul-shaped op has an axis to work on. Probing is a
# yes/no question about expressibility -- the SIZE that a kernel would use is a
# separate decision made from the memory hierarchy (`mechanism.py`).
PROBE_SHAPE = (4, 8)

# Tried in order. Most arithmetic ops take float; the bitwise/shift family
# rejects it, and the logical family wants bool. Trying rather than declaring
# keeps this driven by what the op accepts instead of by a hand-written table
# that would drift from the registry.
PROBE_DTYPES = (torch.float32, torch.int32, torch.bool)


# Dtypes this TARGET cannot express, so an op that decomposes through one is out
# of scope rather than missing an emitter. Reported as its own class because the
# two call for opposite responses: an emitter gap is work, and this is a
# decision. v75 has no complex datapath and no fp64 -- HVX's widest float is 32
# bit and there is no double-precision vector unit, so a "supported" fp64 kernel
# would be scalar `double` software emulation that no accelerator claim could
# ever be made about. Measured on the first full scan: 20 ops decompose through
# complex64 and 6 through float64.
OUT_OF_SCOPE_DTYPES = ("complex64", "complex128", "complex32", "float64")


def _inner_cause(exc, limit=300):
    """The innermost message of a torch exception, ahead of its wrapper.

    torch reports a failed call as "Dynamo failed to run FX node with fake tensors:
    call_function aten.foo(*(FakeTensor(...), ...))" and only THEN the reason it
    actually failed. Truncating the front keeps the wrapper and discards the cause,
    which is exactly backwards: 289 of the 399 rows in the "torch refused" bucket
    recorded nothing but the wrapper, so the bucket could not be audited from the
    artifact at all and had to be re-probed live.
    """
    text = " ".join(str(exc).split())
    m = (re.search(r"got \w*(?:Error|Exception)\((.+)", text)
         or re.search(r"(?:RuntimeError|ValueError|TypeError|NotImplementedError|"
                      r"AssertionError): (.+)", text))
    if m:
        return m.group(1).strip().strip("'\"")[:limit]
    return text[:limit]


class Unprobeable(Exception):
    """This module cannot synthesise a call to the op. NOT a statement about the
    pipeline -- see the module docstring's three-way split."""


def required_args(row) -> list:
    """The arguments a caller must supply: no default, not kwonly.

    Optional and defaulted arguments are omitted deliberately. `aten::clamp`
    takes `Scalar? min = None` and `Scalar? max = None`, and calling it with
    neither is a legitimate call that the registry itself defines; inventing
    values for them would probe an op we were not asked about.
    """
    return [a for a in row.get("args", [])
            if a.get("default") is None and not a.get("kwonly")]


def probe_signature(row):
    """`"unary"`, `"binary"`, or None if this row's required arguments are not a
    shape we can synthesise.

    Restricted on purpose to all-Tensor required arguments returning one Tensor.
    That is the family whose call needs no invented values, so a failure to
    build one is always a fact about the pipeline and never about the prober.
    Everything else is reported as `unprobeable`, with the count, which is how
    the next extension of this module gets prioritised.
    """
    args = required_args(row)
    returns = row.get("returns") or []
    if len(returns) != 1 or not returns[0].get("is_tensor"):
        return None
    if returns[0].get("type") != "Tensor":       # Tensor[] / (Tensor, Tensor)
        return None
    if not args or len(args) > 2:
        return None
    if not all(a.get("is_tensor") and a.get("type") == "Tensor" and
               not a.get("is_list") and not a.get("is_mutable") for a in args):
        return None
    return "unary" if len(args) == 1 else "binary"



# ---- synthesised arguments, and why they are counted separately ---------------
#
# `probe_signature` above accepts only all-Tensor required arguments, because a
# call built from nothing but tensors needs no invented values and so a failure is
# always a fact about the PIPELINE. Most of the harvest is not like that: measured
# over the 687 unprobeable eligible rows, 151 differ only by having 3 or 4 required
# arguments, 121 by taking a Scalar or an int, and 63 by returning a tuple that the
# pipeline has handled since batch 7.
#
# Those are reachable if values are invented for the non-tensor arguments. Doing so
# COSTS the invariant: a row that comes back `blocked` might be blocked because the
# guessed `dim` was out of range rather than because an emitter is missing. The
# answer is not to refuse, and not to pretend -- it is to report the two
# populations separately. `covered` keeps its original meaning; `covered_synth` is
# the weaker claim, and the report prints both.
#
# Values are chosen by argument NAME first and type second, because the name is
# what carries the constraint: `dim` must index the probe shape, `k` must not
# exceed it, and a bare `Scalar` operand can be anything away from 0 and 1.
SYNTH_BY_NAME = {
    "dim": 0,
    "dim0": 0,
    "dim1": 1,
    "start": 0,
    "end": 2,
    "step": 1,
    "k": 2,
    "n": 2,
    "correction": 1,
    "p": 2.0,
    "eps": 1e-5,
    "negative_slope": 0.01,
    "min": 0.25,
    "max": 0.75,
    "other": 2.0,
    "alpha": 1.0,
    "beta": 1.0,
    "exponent": 2.0,
    "value": 2.0,
    "keepdim": False,
    "unbiased": True,
    "descending": False,
    "largest": True,
    "sorted": True,
    # `steps` is a COUNT, not an increment: `linspace(start, end, steps)` returns that
    # many points and torch rejects 0. `SYNTH_BY_TYPE["SymInt"]` is 0 -- right for an
    # index, wrong for a length, the same name-vs-type confusion as `p`. Distinct from
    # `step` (singular), already here as 1, which IS an increment.
    "steps": 8,
    # a REPEAT COUNT, likewise a length rather than an index
    "repeats": 2,
    # `n` as a matrix order (eye, linalg_matrix_power) must be positive
    "n_fft": 16,
    # `steps` is a COUNT, not an increment: `linspace(start, end, steps)` returns that
    # many points and torch rejects 0. `SYNTH_BY_TYPE["SymInt"]` is 0, which is right
    # for an index and wrong for a length -- the same name-vs-type confusion as `p`.
    # Distinct from `step` (singular), already here as 1, which IS an increment.
    "steps": 8,
    # groups=0 IS AN INVALID CONVOLUTION and would be what `SYNTH_BY_TYPE["SymInt"]`
    # gives. The name carries the constraint, which is this table's whole premise.
    "groups": 1,
}

SYNTH_BY_TYPE = {
    "int": 0,
    "SymInt": 0,
    "float": 2.0,
    "Scalar": 2.0,
    "bool": False,
    "int[]": [0],
    "SymInt[]": [0],
    "int[1]": [0],
    "int[2]": [1, 1],
    "float[]": [2.0],
    "Scalar[]": [2.0],
    # OPTIONAL SCALARS IN A REQUIRED SLOT TAKE None, for the same reason a `Tensor?`
    # does: the schema says the argument may be absent, and reading that as "cannot be
    # synthesised" confuses an optional argument with an unsupported one.
    "int?": None,
    "bool?": None,
    "ScalarType?": None,
    "float?": None,
    # A REQUIRED dtype argument gets the probe's own float dtype, so the op is asked
    # about the same dtype the rest of the call uses rather than a second one.
    "ScalarType": torch.float32,
}


#: A SIZED int list -- `SymInt[2]`, `int[3]` -- and the optional list forms the
#: `.vec` overloads use. Matched as a pattern rather than enumerated in
#: `SYNTH_BY_TYPE` because the LENGTH is the spatial rank and the VALUE has to fit
#: the probe tensor, so neither can come from a fixed table.
_SIZED_LIST = re.compile(r"^(?:SymInt|int)\[(\d*)\]\??$")
_FLOAT_LIST = re.compile(r"^float\[(\d*)\]\??$")

#: Spatial argument values that are COHERENT with a probe tensor whose spatial
#: extent is `_SPATIAL_EXTENT`. This is the part a fixed table cannot express: a
#: `kernel_size` of [0] is degenerate, an `output_size` larger than the input is an
#: error for adaptive pooling, and a `stride` unrelated to the kernel makes a
#: convolution whose output shape no other argument agrees with. The reason
#: `MAX_SYNTH_ARGS` was capped at 5 was precisely that the convolution-shaped tail
#: "needs a coherent set of values, not independent guesses" -- so the values are
#: derived here from one chosen extent instead of guessed per slot.
_SPATIAL_EXTENT = 8

#: Channels used by `probe_shape` AND by `_weight_shape`, so a convolution's input
#: and weight cannot disagree about the contraction axis.
_PROBE_CHANNELS = 2

#: The argument names whose value must be derived from the probe shape rather than
#: read from `SYNTH_BY_TYPE`. Named explicitly so the spatial path cannot capture an
#: unrelated list argument by accident.
_SPATIAL_NAMES = frozenset((
    "kernel_size", "kernel", "stride", "strides", "padding", "output_padding",
    "dilation", "dilations", "output_size", "scale_factors", "scales",
    "input_size", "input_sizes", "size", "sizes", "shape",
))


def _spatial_list(name, rank, extent=None):
    """A list of `rank` values for spatial argument `name`, fitting `extent`.

    `extent` is threaded from the caller because the MINER picks a different shape
    than the prober: the prober asks "is this expressible" at extent 8, and the miner
    solves for a tier and may land on 256. Reusing the probe's values there would give
    `upsample` an output_size of 4 against a 256-wide input -- a 64x downsample, a
    valid call and a task nobody meant to build. Coherence between the shape and the
    values is the whole reason these are derived rather than tabulated.
    """
    e = extent or _SPATIAL_EXTENT
    if name in ("kernel_size", "kernel"):
        return [2] * rank
    if name in ("stride", "strides"):
        return [2] * rank                      # matches the kernel: no overlap
    if name in ("padding", "output_padding"):
        return [0] * rank
    if name in ("dilation", "dilations"):
        return [1] * rank
    if name == "output_size":
        # must not EXCEED the input for adaptive pooling; a clean factor of the
        # extent keeps the golden exact rather than interpolation-dependent
        return [e // 2] * rank
    if name in ("scale_factors", "scales"):
        return [2.0] * rank
    if name in ("input_size", "input_sizes", "size", "sizes", "shape"):
        return [e] * rank
    return [1] * rank


def spatial_rank(row):
    """The number of SPATIAL dimensions this row's arguments imply, or None.

    Read off the sized list arguments: `output_size: SymInt[2]` means two spatial
    axes, so the input is (N, C, H, W). `avg_pool3d`'s `kernel_size: int[3]` means
    three, so (N, C, D, H, W). A `.vec` overload carries `SymInt[]?` with no length
    and is assumed 2-D, which is what those overloads are for in practice.

    THIS IS THE PREREQUISITE THE OTHER TWO SYNTH FIXES SHARE, and measuring it is
    what showed they were one job. `PROBE_SHAPE` is `(4, 8)` -- rank 2 -- while every
    op in the 67-missing-argument-type bucket and the 72-over-the-argument-cap bucket
    is spatial: adaptive pooling, the upsample/interpolate family, `_convolution`,
    `_conv_depthwise2d`, the `_batch_norm_*` family. Supplying int-list VALUES
    without also choosing a rank-appropriate SHAPE would move those rows from
    "cannot synthesise" to "torch refused the call we built" -- a different bucket
    and the same zero kernels.
    """
    ranks = set()
    for a in required_args(row):
        if a.get("is_tensor"):
            continue
        m = _SIZED_LIST.match(a.get("type") or "") or \
            _FLOAT_LIST.match(a.get("type") or "")
        if not m:
            continue
        n = m.group(1)
        if n:
            ranks.add(int(n))
        elif a.get("name") in ("output_size", "scale_factors", "kernel_size",
                              "stride", "padding", "dilation"):
            ranks.add(2)                       # a `.vec` overload: no declared length
    if not ranks:
        return None
    # one op may carry both `int[2]` and `int[]`; the DECLARED length wins, and a
    # row declaring two different lengths is not coherent so it stays unprobeable
    return min(ranks) if len(ranks) == 1 else None


def probe_shape(row, default=None):
    """The example-tensor shape for `row`: rank-aware for a spatial op, else the flat
    2-D `PROBE_SHAPE`.

    (N, C, ...) with N = 1 and C = 2 -- the smallest shape that is still a genuine
    batch-and-channel layout, since a pooling or convolution op reads the trailing
    `rank` axes as spatial and everything before them as batch/channel.
    """
    r = spatial_rank(row)
    if r is None:
        return default or PROBE_SHAPE
    return (1, _PROBE_CHANNELS) + (_SPATIAL_EXTENT,) * r


#: Argument names that mark an operand of a matrix/batch-matrix CONTRACTION --
#: driven by NAME and ARITY, never by matching the op's own spelling. The
#: registry exposes the same contraction under more than one name (`mm`,
#: `bmm`, `addmm`, `baddbmm`, `sparse_addmm`, ...), and a fixed op-name list
#: would miss whichever of those a future torch adds -- precisely the failure
#: mode already rejected for HMX detection (`mechanism._contracts` keys on the
#: traced primitive, not on a name table). These four spellings are what
#: PyTorch's own schemas use for a matmul-shaped operand, so keying on them is
#: keying on the registry's convention, not on which kernel happens to exist.
#:
#: CONSULTED, not just declared: `_weight_shape` gates its mat1/mat2/batch1/
#: batch2 dispatch on membership in this set before matching the individual
#: name, so editing the set changes which names get a contraction-shaped
#: operand rather than silently doing nothing (the drift the original
#: docstring warned about and the code did not yet enforce).
_CONTRACTION_NAMES = frozenset({"mat1", "mat2", "batch1", "batch2"})


def contraction_tshapes(names, m=4, k=8):
    """Per-argument shapes for a plain (non-synth) contraction call, or None.

    `names` is the row's REQUIRED tensor argument names in schema order --
    always length 1 or 2 here, since this is only consulted from a binary
    (`sig` is `"unary"`/`"binary"`) call site.

    `m`/`k` parameterise the (M, K) operand size -- the PROBE call site below
    always wants the small, fixed `PROBE_SHAPE`-scale defaults (4, 8), while
    `mine._args_for` wants the CURRENT ladder rung's own (M, K) so the sized
    task actually lands at the tier being solved for, not at a fixed toy
    size. NOT a second copy of the geometry: `mine.py` had the identical
    same-shape-for-both-operands defect this function was written to fix
    here first (`mm((8,64),(8,64))` is not a valid matmul), and re-deriving
    the (M,K) x (K,N) shapes a second time in `mine.py` would let the two
    drift out of sync silently -- so `mine._args_for` imports and calls this
    function directly instead.

    PUBLIC (no leading underscore) for exactly that reason: this is no
    longer coverage.py's own implementation detail once `mine.py` calls it.

    Returns a LIST of candidate shape-lists to try in order, because `mm` and
    `bmm` declare the IDENTICAL parameter names (`self`, `mat2`) for a rank-2
    and a rank-3 call respectively -- name alone cannot tell them apart, so
    both are offered and the caller keeps whichever one torch accepts, rather
    than guessing from the op's name which this function never sees.
    """
    names_set = set(names)
    if "mat2" not in names_set or "self" not in names_set or len(names) != 2:
        return None
    # (M, K) x (K, N), M != K != N so a working call cannot be mistaken for a
    # tensor that merely happens to be square.
    rank2 = [(m, k), (k, m)]                       # self=(M,K), mat2=(K,N)
    rank3 = [(2, m, k), (2, k, m)]                 # batched: (B,M,K), (B,K,N)
    return [rank2, rank3] if names[0] == "self" else [rank2[::-1], rank3[::-1]]


def _weight_shape(arg, rank, extent=None, channels=None):
    """A per-position shape for a tensor argument, or None to use the caller's.

    Only the convolution-family `weight` gets one, and it is derived from the same
    channel count `probe_shape` uses so the two agree: (C, C, k...) against an input
    of (1, C, E...). A 3-wide kernel at the synthesised stride 2 and padding 0 gives
    a valid output for E >= 3, and `groups` is 1 so the channel axes match exactly.
    """
    # SOME TENSOR ARGUMENTS MUST BE SCALARS. `linspace.Tensor_Tensor(Tensor start,
    # Tensor end, ...)` rejects anything else -- "linspace only supports 0-dimensional
    # start and end tensors" -- and we were handing it the full probe shape. Same for
    # the `logspace` family. An empty shape IS a valid torch shape (a 0-d tensor), so
    # this is a per-position shape like the convolution weight, not a special case in
    # the caller.
    # An INDEX tensor must be integral -- "expected indices to be long or int, got
    # float". Signalled by returning the sentinel below; the caller draws it as int64.
    if arg.get("name") in ("index", "indices", "target", "sorter") and arg.get("is_tensor"):
        return "int64"
    if arg.get("name") in ("start", "end", "base", "steps") and arg.get("is_tensor"):
        return ()
    if rank and arg.get("name") in ("weight", "filter", "filters"):
        # THE CHANNEL COUNT COMES FROM THE CHOSEN SHAPE, not from a constant. The
        # prober uses `_PROBE_CHANNELS` because it picks the input shape itself; the
        # MINER walks a ladder whose channel counts are 4, 8, 16, 32..., so a weight
        # built at the constant has a Cin the input does not have and torch rejects the
        # call. That presented as "convolution cannot be sized to any tier".
        #
        # Fourth instance this session of one bug class: two synthesised values that
        # must AGREE, produced independently. The others were the probe shape vs the
        # spatial rank, the weight's kernel extent vs `kernel_size`, and the tensor
        # shapes in `coverage.probe` vs `mine._args_for`.
        c = channels or _PROBE_CHANNELS
        # THE KERNEL EXTENT MUST MATCH `_spatial_list("kernel_size")`, and it did not.
        # This hardcoded 3 while `kernel_size` synthesised to [2, 2]. `convolution` has
        # no `kernel_size` argument -- the weight IS the kernel -- so the two could
        # never disagree there and the bug was invisible. `slow_conv_transpose2d`,
        # `slow_conv3d` and the `thnn_conv*` family all take BOTH, and torch rejects a
        # weight whose trailing extents contradict the kernel_size it was handed.
        #
        # Derived from the same helper rather than repeated, so a change to one cannot
        # desynchronise them again -- the same argument as `_PROBE_CHANNELS` being
        # shared with `probe_shape`.
        k = _spatial_list("kernel_size", rank)
        return (c, c) + tuple(k)
    # CONTRACTION OPERANDS -- `addmm`-shaped (`self, mat1, mat2`) and
    # `baddbmm`-shaped (`self, batch1, batch2`) rows reach this synth path
    # because they have 3+ required tensors, so `probe_signature` returns
    # `None` before `contraction_tshapes` (the 1-2-tensor path) ever runs.
    # NAME-driven like every other case here: any row exposing these
    # parameter names is treated as a contraction, whatever the op is called.
    #
    # `self` is deliberately left untouched (falls through to `None`, i.e.
    # the caller's ordinary shape). It is not one of the two matmul operands
    # here -- it is the accumulator `beta*self + alpha*(mat1 @ mat2)` is
    # added to -- so mat1/mat2 (and batch1/batch2) are sized to PRODUCE a
    # (4, 8) result: whatever shape the caller already uses for `self`
    # (`probe`'s PROBE_SHAPE) is only actually compatible at THAT fixed probe
    # scale -- see `_CONTRACTION_NAMES`'s own note below; the miner's ladder
    # rung is a DIFFERENT, larger `self` shape and this branch does not (yet)
    # scale mat1/mat2/batch1/batch2 to match it, so addmm/baddbmm currently
    # size only at the probe scale, not across the miner's tier ladder --
    # flagged, not fixed, this round.
    #
    # GATED ON `_CONTRACTION_NAMES` membership, not a fourth repetition of
    # the four literal strings: editing that set now actually changes which
    # argument names this branch special-cases, closing the exact drift its
    # own comment warned about (`mine.py`'s task-1b report flagged the
    # frozenset as declared and never consulted).
    if arg.get("name") in _CONTRACTION_NAMES:
        if arg.get("name") == "mat1":
            return (4, 8)
        if arg.get("name") == "mat2":
            return (8, 8)
        if arg.get("name") == "batch1":
            return (2, 4, 8)
        if arg.get("name") == "batch2":
            return (2, 8, 8)
    return None


#: (op, argument) pairs whose value the NAME alone gets wrong.
#:
#: `SYNTH_BY_NAME["p"] = 2.0` is right for `norm` and `pow` -- an exponent -- and wrong
#: for every op where `p` is a PROBABILITY, which torch rejects outright:
#: "geometric_ expects p to be in (0, 1), but got p=2.0". The name carries the
#: constraint for most arguments and not for this one, so the exception is keyed on the
#: PAIR rather than weakening the name table for the ops it already serves correctly.
#: Arguments that must be ABSENT rather than filled. Keyed by (op-suffix, name) and
#: matched on the overload, because it is the `.vec` FORM that is exclusive:
#: "Must specify exactly one of output_size and scale_factors". We were filling both,
#: which torch refuses outright -- eleven rows, the whole `_aa.vec` and
#: `upsample_*.vec` family.
SYNTH_ABSENT_BY_OVERLOAD = {("vec", "scale_factors")}

SYNTH_BY_OP_AND_NAME = {
    # `repeat` tiles the tensor, so its list is one entry PER DIMENSION and torch
    # refuses a shorter one: "Number of dimensions of repeat dims can not be smaller
    # than number of dimensions of tensor". `SYNTH_BY_TYPE["int[]"] = [0]` is length 1
    # -- fine for a `dim` list, wrong here, and 0 would produce an empty tensor anyway.
    # Two entries of 2 tile a rank-2 probe into a 2x block, which is what the op is for.
    ("repeat", "repeats"): [2, 2],
    ("tile", "dims"): [2, 2],
    ("geometric", "p"): 0.5,
    ("bernoulli", "p"): 0.5,
    ("dropout", "p"): 0.5,
    ("feature_dropout", "p"): 0.5,
    ("alpha_dropout", "p"): 0.5,
    ("feature_alpha_dropout", "p"): 0.5,
    ("native_dropout", "p"): 0.5,
    ("binomial", "p"): 0.5,
}


def synth_value(arg, rank=None, extent=None, *, op=None, overload=None):
    """A plausible value for one non-tensor required argument, or raise.

    Name before type: `dim` has to index the probe shape and `k` has to fit it,
    while a bare `Scalar` operand only has to be away from 0 and 1 (several ops are
    degenerate there). An argument this cannot fill leaves the row unprobeable,
    which keeps the "prober cannot call it" bucket meaningful.

    `rank` is the row's spatial rank when it has one, and it is what lets a SIZED
    list argument be filled at all -- the length is the rank and the values have to
    fit the probe tensor, so neither is expressible as a table entry. Without a rank
    a sized list still raises, so a non-spatial row's behaviour is unchanged.
    """
    name, typ = arg.get("name"), arg.get("type")
    # KEYWORD-ONLY `op`, deliberately. A previous attempt added it as a fourth
    # POSITIONAL parameter and a full re-probe lost 21 expressible ops; making it
    # keyword-only removes that whole class of caller confusion rather than auditing
    # every call site for arity.
    if (overload, name) in SYNTH_ABSENT_BY_OVERLOAD:
        return None
    if op is not None and (op, name) in SYNTH_BY_OP_AND_NAME:
        return SYNTH_BY_OP_AND_NAME[(op, name)]
    if name in SYNTH_BY_NAME:
        return SYNTH_BY_NAME[name]
    # BEFORE `SYNTH_BY_TYPE`, and the order is load-bearing. That table carries
    # `"int[2]": [1, 1]`, which shadowed every spatial list and returned a
    # DEGENERATE value -- a 1x1 pooling kernel with stride 1 is the identity, and a
    # convolution so shaped tests nothing. It also made the no-rank case return a
    # value instead of raising, so a non-spatial row silently got a spatial guess.
    # Both were caught by asserting the values rather than assuming them.
    m = _SIZED_LIST.match(typ or "") or _FLOAT_LIST.match(typ or "")
    if m and rank and name in _SPATIAL_NAMES:
        n = int(m.group(1)) if m.group(1) else rank
        return _spatial_list(name, n, extent)
    if typ in SYNTH_BY_TYPE:
        return SYNTH_BY_TYPE[typ]
    if m and rank:
        return _spatial_list(name, int(m.group(1)) if m.group(1) else rank, extent)
    raise Unprobeable(f"no synthesised value for {name!r}: {typ}")


def _is_plain_tensor_return(ret) -> bool:
    """Is this return a plain `Tensor` by VALUE -- ignoring a return NAME, refusing
    an alias marker or a list?

    THE NAME IS NOT PART OF THE TYPE, and treating it as one was a bug that hid 61
    ops. `synth_signature` used to test `r["type"] == "Tensor"`, and PyTorch NAMES
    the returns of a multi-result schema:

        aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
        aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
        aten::aminmax(Tensor self, ...) -> (Tensor min, Tensor max)

    so the type string arrived as `"Tensor values"`, the equality failed, and the row
    was filed as "the prober cannot synthesise a call" -- a statement about the OP
    when it was a statement about our string comparison. Two ops were refused for a
    NAME on a SINGLE return (`cudnn_grid_sampler` -> `Tensor output`).

    It contradicted the caller's own docstring, which says multiple plain-Tensor
    returns are accepted and that the pipeline has handled them since batch 7. The
    proof it was a bug and not a policy: `aminmax`, `cummax`, `sort` and `topk` are
    all in the refused set, and all four are ALREADY hand-built kernels in this
    corpus (`fp32_aminmax`, `fp32_cummax_rows`, `fp32_sort_rows`, `fp32_topk_rows`).
    The pipeline could build them; only the MINER could not see them.

    WHAT IS STILL REFUSED, and it is the majority of what looked blocked:
      * `Tensor(a)` -- an ALIASING return, i.e. a view. The pipeline emits value
        semantics, so a view is not a kernel. 93 rows, correctly excluded.
      * `Tensor[]`, `Tensor[](*)`, `(Tensor, t)`, `Dict(Tensor, t)` -- a list or
        container return, which is a different emitter.
      * a non-tensor return (`SymInt max_q` in the flash-attention schemas), which
        is why those stay out even though their other returns are tensors.
    Only the leading token is inspected for the marker, so `Tensor(a)` and
    `Tensor(a!) foo` both fail while `Tensor foo` passes.
    """
    head = (ret.get("type") or "").split(" ", 1)[0]
    return head == "Tensor"


#: RAISED FROM 5 TO 14, and only because the two things the cap was standing in for
#: are now handled explicitly. The old comment said the tail "needs a coherent set of
#: values, not independent guesses" -- true, and the cap was a proxy for it. The values
#: are now derived from one extent (`_spatial_list`) and the SHAPES from one input
#: (`_weight_shape`), so arity by itself is no longer evidence of incoherence.
#: `convolution` has 9 required arguments and `_convolution` 13.
MAX_SYNTH_ARGS = 14
MAX_SYNTH_TENSORS = 4


#: Synthesised probing is OFF by default, and the reason is a measured crash.
#:
#: The extension makes 341 of the 1,170 eligible (op, overload) pairs callable that
#: `probe_signature` alone cannot reach -- a large gain. It also SEGFAULTS the
#: interpreter on the full harvest. The rows it newly reaches include
#: `_ctc_loss`, `_cholesky_solve_helper`, `_convert_weight_to_int4pack` and
#: `_dyn_quant_matmul_4bit`: kernels with real preconditions among their arguments,
#: handed independent guesses. A bad value there does not raise, it takes the
#: process down, and a prober that dies cannot produce a number at all.
#:
#: Making it safe means running each probe in its own process, which is a design
#: change (1,170 subprocess spawns, and a crash has to be attributed to a row
#: rather than lost). Until then the default scan behaves exactly as it did before
#: this extension, and `--allow-synth` opts in for anyone measuring a narrower set.
ALLOW_SYNTH_DEFAULT = False


def synth_signature(row, extent=None, channels=None):
    """A per-position argument PLAN for a row needing invented values, or None.

    Returns `[("t", None) | ("v", value), ...]` in argument order, so tensor and
    non-tensor arguments may INTERLEAVE. The first version of this function
    required tensors to lead and capped them at two, which rejected the largest
    bucket it existed for: the 98 eligible rows with three required arguments --
    `where.self`, `addmm`, `clamp.Tensor`, `baddbmm`. It reached three extra ops
    out of 250.

    Still refused, and each for a reason:

      * more than MAX_SYNTH_ARGS arguments -- the tail of the registry is
        convolution-shaped and needs a coherent set of values, not independent
        guesses;
      * a mutable argument -- an in-place op has no value semantics to emit;
      * a Tensor[] argument -- `cat`-shaped ops need a list whose members have to
        agree on shape, which is a different synthesiser;
      * a return that is not plain `Tensor` -- `Tensor(a)` is an ALIASING return,
        i.e. a view, and the pipeline emits value semantics. Multiple plain-Tensor
        returns ARE accepted: the pipeline has handled them since batch 7.
    """
    args = required_args(row)
    returns = row.get("returns") or []
    if not args or len(args) > MAX_SYNTH_ARGS:
        return None
    # The row's spatial rank, if it has one: what makes a SIZED list argument
    # fillable, and what `probe_shape` uses to pick a matching tensor rank. The two
    # must agree -- a [2]-long kernel over a 2-D tensor is not a call torch accepts.
    rank = spatial_rank(row)
    if not returns or not all(_is_plain_tensor_return(r) for r in returns):
        return None
    plan, ntens = [], 0
    for a in args:
        if a.get("is_mutable"):
            return None
        if a.get("is_tensor") and a.get("type") == "Tensor" and not a.get("is_list"):
            ntens += 1
            # PER-POSITION SHAPE, not one shape for every tensor. A convolution's
            # weight is (O, C, kh, kw) and its input is (N, C, H, W) -- the same
            # rank and NOT the same shape, with a channel axis that must AGREE.
            # Drawing both at one shape is what made arity look like the problem.
            plan.append(("t", _weight_shape(a, rank, extent, channels)))
            continue
        if a.get("is_tensor") and not a.get("is_list") and                 (a.get("type") or "").endswith("?"):
            # AN OPTIONAL TENSOR IN A REQUIRED SLOT IS FILLED WITH None, which is a
            # call the schema itself defines -- `convolution(..., bias=None, ...)` is
            # an ordinary bias-free convolution. Refusing it treated "this argument
            # may be absent" as "this argument cannot be synthesised".
            plan.append(("v", None))
            continue
        if a.get("is_tensor"):
            return None                 # Tensor[] in a required slot
        try:
            plan.append(("v", synth_value(a, rank, extent, op=row.get("op"),
                                          overload=row.get("overload"))))
        except Unprobeable:
            return None
    if ntens == 0 or ntens > MAX_SYNTH_TENSORS:
        return None
    return plan


def _overload(row):
    ov = row.get("overload") or ""
    return ov if ov else "default"


def _packet(row):
    """The `torch.ops.aten` callable for a row, or raise `Unprobeable`."""
    if row.get("namespace") != "aten":
        raise Unprobeable(f"namespace {row.get('namespace')!r} is not aten")
    try:
        packet = getattr(torch.ops.aten, row["op"])
        return getattr(packet, _overload(row))
    except (AttributeError, RuntimeError) as e:
        raise Unprobeable(f"not reachable through torch.ops.aten: {e}") from e


class _Call(torch.nn.Module):
    """A module whose forward is exactly one registry call.

    `trace()` takes a module, and one op per module is what makes the result
    attributable: every primitive in the traced graph came from this op's
    decomposition and nothing else.
    """

    def __init__(self, fn, synth_plan=None):
        super().__init__()
        self.fn = fn
        # A SYNTHESISED SCALAR IS A CONSTANT, NOT A GRAPH INPUT, and baking it here
        # is what makes that true. `trace()` reads `.shape` off every example
        # argument, so passing a float in the argument tuple failed with
        # "'float' object has no attribute 'shape'" -- 160 rows, every `.Scalar`
        # overload among them, reported as unprobeable for a reason that was ours
        # rather than a fact about the op.
        #
        # Baking also gives the right ANSWER, not just a working call: only tensors
        # become placeholders, so the emitted kernel's signature is the tensor list
        # and the scalar appears as a literal in the C -- which is what it is.
        self.synth_plan = synth_plan

    def forward(self, *args):
        if self.synth_plan is None:
            return self.fn(*args)
        it = iter(args)
        return self.fn(*(next(it) if kind == "t" else v
                         for kind, v in self.synth_plan))


def probe(row, shape=None, allow_synth=ALLOW_SYNTH_DEFAULT):
    """Try to express one harvested row. Returns (status, detail).

    status is "covered", "blocked" or "unprobeable"; detail is the sorted list of
    missing primitive targets for "blocked", and a reason string otherwise.
    """
    # RANK-AWARE unless the caller pinned a shape. A spatial op reads its trailing
    # axes as spatial, so probing `avg_pool3d` at (4, 8) fails in torch for a reason
    # that says nothing about whether the pipeline can express it.
    shape = shape if shape is not None else probe_shape(row)
    sig = probe_signature(row)
    plan = None
    suffix = ""
    if sig is None:
        if not allow_synth:
            return "unprobeable", ("required arguments are not 1-2 plain Tensors "
                                   "-> Tensor")
        plan = synth_signature(row)
        if plan is None:
            return "unprobeable", ("required arguments are not 1-2 plain Tensors "
                                   "-> Tensor, and cannot be synthesised")
        n = sum(1 for kind, _v in plan if kind == "t")
        suffix = "_synth"
    else:
        n = 1 if sig == "unary" else 2

    fn = _packet(row)
    mod = _Call(fn, plan)

    # PER-POSITION SHAPES. `plan` may carry a shape on a tensor entry (a convolution
    # weight), so the tensor list is built from the PLAN rather than from `n` copies of
    # one shape. Without a plan every tensor keeps the single probe shape, exactly as
    # before.
    #
    # CONTRACTION-SHAPED CANDIDATES, base (non-synth) path only. `mm`/`bmm` have
    # exactly 2 required plain-Tensor arguments, so `sig` is `"binary"` (not `None`)
    # and they NEVER reach the synth branch above -- probing them at one shape for
    # both operands (`(4, 8)` against `(4, 8)`) is invalid for a matmul (`mm` needs
    # `self @ mat2` with `self`'s last dim == `mat2`'s first) and reported them
    # `unprobeable` for a reason that was the prober's, not the op's.
    # `contraction_tshapes` returns more than one candidate ONLY for this case,
    # because `mm` (rank 2) and `bmm` (rank 3) declare the identical parameter names
    # -- the ambiguity is resolved by which shape torch accepts below, never by
    # matching the op's own name. PUBLIC function, no leading underscore: `mine.py`
    # calls it too (at its own ladder-rung scale), rather than re-deriving this
    # geometry a second time.
    tshape_candidates = None
    if plan is None:
        names = [a["name"] for a in required_args(row) if a.get("is_tensor")]
        if len(names) == n:
            tshape_candidates = contraction_tshapes(names)
    if tshape_candidates is None:
        tshape_candidates = [[shape] * n]
    if plan is not None:
        # `sh is not None`, NOT `sh or shape`. AN EMPTY TUPLE IS FALSY, so a
        # 0-DIMENSIONAL per-position shape -- exactly what
        # `linspace(Tensor start, Tensor end)` requires -- was read as "no shape given"
        # and replaced by the probe shape. The op then failed with "linspace only
        # supports 0-dimensional start and end tensors" and was filed as a fact about
        # the OPERATOR. Six rows.
        tshape_candidates = [[(sh if sh is not None else shape)
                              for kind, sh in plan if kind == "t"]]

    errors = []
    for tshapes in tshape_candidates:
        for dtype in PROBE_DTYPES:
            if dtype is torch.bool:
                args = tuple(torch.ones(s_, dtype=dtype) for s_ in tshapes)
            elif dtype.is_floating_point:
                # away from 0 and 1: several ops (log, rsqrt, div) are undefined or
                # degenerate there, and a NaN golden would fail for a reason that has
                # nothing to do with whether the op is expressible.
                args = tuple(torch.rand(s_, dtype=dtype) + 0.5 for s_ in tshapes)
            else:
                args = tuple(torch.randint(1, 5, s_, dtype=dtype) for s_ in tshapes)
            # NOT interleaved into `args`: the module bakes the plan's values, so only
            # the tensors are passed and only they become graph placeholders. See _Call.
            try:
                g = trace(mod, args, f"probe_{row['op']}")
            except Exception as e:                 # noqa: BLE001 - registry is wide
                errors.append(f"{dtype}: {type(e).__name__}: {_inner_cause(e)}")
                continue
            bad = sorted({d for d in _graph_dtypes(g) if d in OUT_OF_SCOPE_DTYPES})
            if bad:
                return "out_of_scope", f"decomposes through {', '.join(bad)}"
            try:
                emit_c(g)
            except UnsupportedPrimitive as e:
                missing = sorted({t for t in _missing_targets(g)})
                return "blocked" + suffix, missing or [str(e)[:120]]
            except Exception as e:                 # noqa: BLE001
                return "blocked" + suffix, [f"emit failed: {type(e).__name__}: {str(e)[:120]}"]
            # ANNOTATE TOO, and this is not optional. Emitting C is one of the build
            # stages; `plan_for` needs the schedule annotation, and a primitive can
            # have an emitter and no schedule rule. Checking only the emitter reported
            # 99 covered ops of which batch 4's own `gt.Tensor` was one -- it emitted
            # C perfectly and then died at annotate with no tier. A coverage number
            # that a chosen kernel then fails to build is worse than no number.
            try:
                annotate(g)
            except UnschedulablePrimitive:
                missing = sorted({t for t in _unschedulable_targets(g)})
                return "blocked", [f"no schedule rule: {t}" for t in missing]
            except Exception as e:                 # noqa: BLE001
                return "blocked", [f"annotate failed: {type(e).__name__}: {str(e)[:120]}"]
            return "covered" + suffix, f"{sig}, {dtype}"
    return "unprobeable", "; ".join(errors[:3])


def _graph_dtypes(graph) -> list:
    """Every dtype appearing anywhere in the graph, inputs and intermediates.

    Intermediates matter as much as the signature: `aten::abs` on a complex
    input has a REAL output, so looking only at the result would call it
    expressible while the graph carries a complex intermediate nothing on this
    target can hold.
    """
    return [n.dtype for n in list(graph.inputs) + list(graph.nodes)
            if n.dtype is not None]


def _unschedulable_targets(graph) -> list:
    """Every primitive `schedule_of` refuses. All of them, for the same reason
    `_missing_targets` collects all missing emitters: the histogram is only
    meaningful if it counts every piece of work, not the first one hit."""
    shapes = {n.name: n.shape for n in list(graph.inputs) + list(graph.nodes)}
    out = []
    for n in graph.nodes:
        try:
            schedule_of(n, shapes)
        except Exception:                          # noqa: BLE001
            out.append(n.target)
    return out


def _missing_targets(graph) -> list:
    """Every primitive in `graph` with no emitter.

    ALL of them, not just the first. `emit_c` stops at the one it hit, but a
    graph blocked on three primitives is three pieces of work, and the histogram
    this feeds is only meaningful if it counts them all.
    """
    from hexkernels.forge.frontend.primitives import EMITTERS
    return [n.target for n in graph.nodes if n.target not in EMITTERS]


def scan(rows=None, harvest_path=DEFAULT_HARVEST, limit=None,
         allow_synth=ALLOW_SYNTH_DEFAULT):
    """Probe every eligible row. Returns a result dict keyed by status."""
    if rows is None:
        rows = load_harvest(harvest_path)
    eligible = [r for r in rows if mechanism_eligible(r)]

    # One row per (op, overload): the harvest can carry the same schema from more
    # than one accessor, and coverage is a property of the op, not of the row.
    seen, uniq = set(), []
    for r in eligible:
        key = (r["op"], _overload(r))
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    if limit:
        uniq = uniq[:limit]

    # `covered`/`blocked` mean "reached with NO invented argument values"; the
    # `_synth` twins mean the prober had to make some up. Kept apart because the
    # whole point of this module's split is that a `blocked` row is a fact about
    # the pipeline -- once values are synthesised, it might instead be a fact
    # about the guess.
    out = {"covered": {}, "blocked": {}, "unprobeable": {}, "out_of_scope": {},
           "covered_synth": {}, "blocked_synth": {}, "total": len(uniq)}
    for row in uniq:
        key = f"{row['op']}.{_overload(row)}"
        try:
            status, detail = probe(row, allow_synth=allow_synth)
        except Unprobeable as e:
            status, detail = "unprobeable", str(e)
        except Exception as e:                     # noqa: BLE001
            status, detail = "unprobeable", f"{type(e).__name__}: {str(e)[:120]}"
        out[status][key] = detail
    return out


def missing_histogram(result) -> list:
    """(primitive, how many harvested ops it blocks), most blocking first.

    THE ACTIONABLE OUTPUT. An emitter is worth writing in proportion to the
    number of harvested ops it unblocks, and that number is measured here rather
    than guessed from how common the op feels.
    """
    counter = collections.Counter()
    for bucket in ("blocked", "blocked_synth"):
        for missing in result.get(bucket, {}).values():
            for target in missing:
                counter[target] += 1
    return counter.most_common()


def render(result, missing_top=15) -> str:
    total = result["total"]
    cs = len(result.get("covered_synth", {}))
    bs = len(result.get("blocked_synth", {}))
    cov, blk, unp, oos = (len(result.get(k, {})) for k in
                          ("covered", "blocked", "unprobeable", "out_of_scope"))
    lines = [
        "# Harvest coverage",
        "",
        f"eligible ops probed : {total}",
        f"  covered           : {cov}   ({100.0 * cov / total:.1f}%) traced + emitted",
        f"  blocked           : {blk}   ({100.0 * blk / total:.1f}%) traced, missing an emitter",
        f"  out of scope      : {oos}   ({100.0 * oos / total:.1f}%) needs a dtype this target has no datapath for",
        f"  covered (synth)   : {cs}   ({100.0 * cs / total:.1f}%) reached only by "
        f"INVENTING argument values",
        f"  blocked (synth)   : {bs}   ({100.0 * bs / total:.1f}%) same, and an "
        f"emitter is missing",
        f"  unprobeable       : {unp}   ({100.0 * unp / total:.1f}%) prober cannot call it",
        "",
        "Covered means SCALAR C WAS EMITTED -- not that it is correct. Only the",
        "simulator and the generated golden harness decide that (`run_batch`).",
        "",
        f"## Emitter targets, ranked by ops unblocked (top {missing_top})",
        "",
    ]
    hist = missing_histogram(result)
    if not hist:
        lines.append("(nothing blocked)")
    for target, count in hist[:missing_top]:
        lines.append(f"  {count:4d}  {target}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest", default=str(DEFAULT_HARVEST))
    ap.add_argument("--limit", type=int, default=None,
                    help="probe only the first N eligible ops (for a quick look)")
    ap.add_argument("--missing", type=int, default=15,
                    help="how many ranked emitter targets to print")
    ap.add_argument("--show-covered", action="store_true",
                    help="list the ops a batch can draw on today")
    ap.add_argument("--json", default=None, help="write the full result here")
    args = ap.parse_args(argv)

    result = scan(harvest_path=args.harvest, limit=args.limit)
    print(render(result, missing_top=args.missing))
    if args.show_covered:
        print("\n## Covered ops\n")
        for key in sorted(result["covered"]):
            print(f"  {key:44s} {result['covered'][key]}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, sort_keys=True)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
