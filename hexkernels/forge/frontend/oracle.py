"""Golden vectors + a generated C++ harness that checks a candidate kernel
against them.

Until this module (and `roundtrip.py` next to it) exist, nothing proves the
C emitted by `emit.py` (graph.py) is *correct* -- only that it looks right.
`golden()` runs the SAME traced module (via its own `export`+decompositions,
not `module.forward`) on seeded random inputs covering every graph
placeholder -- including the parameters `torch.export` lifts ahead of the
activations (see `trace.py`) -- so the values fed to the emitted kernel and
the values used to compute the expected outputs are, by construction, the
same numbers. `harness_c()` turns one `golden()` dict into a self-contained
`main()` that calls `candidate_kernel`, compares, and reports a
machine-parseable verdict.

Seeded generation always goes through an explicit `torch.Generator` --
never global RNG state -- so the same module and seed reproduce the same
harness byte for byte (see `roundtrip.py`'s determinism tests).
"""
from typing import Any

import torch
from torch.export import export

from hexkernels.core import target as _target
from hexkernels.forge.frontend.graph import DTYPE_C
from hexkernels.forge.frontend.trace import decomp_table, trace

# torch dtype for every DTYPE_C key this stage can generate golden data for.
_TORCH_DTYPE = {
    "float32": torch.float32,
    "int32": torch.int32,
    "int8": torch.int8,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "float16": torch.float16,
}

# Integer generation ranges. Wide enough to exercise a kernel's arithmetic
# (shifts, adds, clamps) without overflowing the dtype or the int32
# accumulation the pilot's int8 chain performs before its final cast.
_INT_RANGE = {
    "int8": (-100, 100),
    "uint8": (0, 200),
    "int32": (-1000, 1000),
    "int64": (-1000, 1000),
}


def _seeded_tensor(shape: tuple, dtype_str: str, gen: torch.Generator) -> torch.Tensor:
    """One tensor of `shape`/`dtype_str`, drawn from the explicit generator
    `gen` -- never `torch.manual_seed`/global RNG, so callers can seed once
    per `golden()` call and get the same tensor for the same (shape, dtype,
    position-in-sequence) every time."""
    torch_dtype = _TORCH_DTYPE[dtype_str]
    if dtype_str in ("float32", "float16"):
        # Generate in float32 (CPU randn with a generator is most reliably
        # supported there) then narrow -- exact for float32, standard
        # round-to-nearest for float16.
        return torch.randn(shape, generator=gen, dtype=torch.float32).to(torch_dtype)
    lo, hi = _INT_RANGE[dtype_str]
    return torch.randint(lo, hi, shape, generator=gen, dtype=torch_dtype)


def golden(module: torch.nn.Module, example_args: tuple, seed: int = 0) -> dict:
    """Seeded golden vectors for `module`, in graph placeholder order.

    Returns ``{"inputs": [np.ndarray, ...], "outputs": [np.ndarray, ...]}``.
    `inputs` covers EVERY placeholder `trace()` records -- parameters the
    exporter lifted ahead of activations included -- because those are
    exactly the arguments the emitted `candidate_kernel` takes. Outputs are
    computed by calling the traced-and-decomposed graph module directly with
    those same seeded values (not `module.forward`, which would use the
    module's real trained parameters instead of the seeded ones and make
    `inputs`/`outputs` inconsistent with each other).
    """
    module = module.eval()
    g = trace(module, example_args, name="oracle_golden")

    ep = export(module, example_args).run_decompositions(decomp_table())
    graph_module = ep.graph_module

    gen = torch.Generator().manual_seed(seed)

    # THE SPEC'S ARGUMENT VALUES ARE THE TASK, and this used to ignore them.
    #
    # `example_args` was consumed for SHAPE and DTYPE only; every value came from
    # `_seeded_tensor`, i.e. `randn` for floats and `randint(_INT_RANGE)` for
    # integers. So a KernelSpec that carefully drew its divisor from [0.5, 1.5] or
    # planted a NaN or bounded a Q15 multiplier got none of it -- the golden was
    # standard normal regardless, and the stated range was fiction.
    #
    # Found by mining: `acos`, `acosh`, `arcsin`, `arctanh`, `log` and `log10` all
    # failed their own references, at 32%, 85%, 32%, 32%, 50% and 50% of elements
    # -- exactly the fraction of a standard normal that falls outside each op's
    # DOMAIN. The domain shift the mined specs applied was dead code.
    #
    # Values are taken from `example_args` when they correspond one-to-one with the
    # placeholders. They may not: `trace` records parameters the exporter lifted
    # ahead of the activations, and those have no example_arg. In that case each
    # placeholder is matched by position among the trailing arguments and anything
    # unmatched still falls back to the seeded draw, which is what keeps a module
    # with real parameters working.
    supplied = list(example_args)
    inputs = []
    lifted = len(g.inputs) - len(supplied)
    for i, node in enumerate(g.inputs):
        j = i - lifted
        if 0 <= j < len(supplied):
            t = supplied[j]
            if (isinstance(t, torch.Tensor)
                    and tuple(t.shape) == tuple(node.shape)
                    and str(t.dtype).replace("torch.", "") == node.dtype):
                inputs.append(t.detach().clone())
                continue
        inputs.append(_seeded_tensor(node.shape, node.dtype, gen))

    # AN INDEX TENSOR CANNOT BE SEEDED LIKE A VALUE TENSOR. `_INT_RANGE` gives int64
    # placeholders values in [-1000, 1000), which for a gather's index operand is
    # out of bounds -- `index_select` raises and the whole task fails to build. The
    # bound is not a property of the tensor, it is a property of the op that
    # consumes it, so it is read off the graph: for `index_select(x, dim, idx)` the
    # valid range is [0, x.shape[dim]).
    #
    # Clamped rather than re-drawn so the values stay a deterministic function of
    # the seed, and so a placeholder feeding two different gathers gets the tighter
    # bound.
    by_name = {n.name: i for i, n in enumerate(g.inputs)}
    for node in g.nodes:
        if node.target != "aten.index_select.default":
            continue
        src, dim, idx = node.args[0], node.args[1], node.args[2]
        if idx not in by_name:
            continue
        src_shape = next((n.shape for n in g.inputs if n.name == src), None)
        if src_shape is None:
            src_shape = next((n.shape for n in g.nodes if n.name == src), None)
        if not src_shape:
            continue
        limit = src_shape[dim % len(src_shape)]
        k = by_name[idx]
        inputs[k] = inputs[k].abs() % limit

    with torch.no_grad():
        raw_outputs = graph_module(*inputs)
    if isinstance(raw_outputs, torch.Tensor):
        raw_outputs = (raw_outputs,)

    return {
        "inputs": [t.detach().contiguous().numpy() for t in inputs],
        "outputs": [t.detach().contiguous().numpy() for t in raw_outputs],
        # Per output, the accumulated magnitude and term count when the output IS
        # an accumulation -- the scale a reduction's error is really set by. See
        # `accum_scales`. `(None, 0)` for everything else, which keeps the harness
        # for a non-reduction kernel byte-identical to what it was.
        "accum": accum_scales(graph_module, inputs),
    }


# Outputs whose value IS an accumulation, and how to bound the accumulated
# magnitude of each element. See `accum_scales` for why this exists.
#
# `mm` is included even though no matmul in this corpus currently needs it: the
# argument applies to any accumulation, and the extra term is proportional to
# n*u*S, which is negligible unless the sum cancels. A rule that only fires where
# it was discovered is a special case; this one is uniform.
#
# Convolution is NOT included. Its accumulated magnitude is computable (an abs
# convolution) but the padding and dilation bookkeeping is a second implementation
# of the reference, and no conv output has needed it. Recorded as a gap.
_ACCUM_SCALE = {
    "aten.sum.dim_IntList": "reduce_dims",
    "aten.mean.dim": "reduce_dims",
    "aten.prod.dim_int": None,          # a product's error is not bounded this way
    "aten.mm.default": "matmul",
}


class _Recorder(torch.fx.Interpreter):
    """Runs the decomposed graph and keeps every node's value.

    `golden()` already executes this graph to produce the expected outputs; this
    subclass keeps the intermediates so the tolerance can be computed from the
    values that were actually accumulated, rather than from the result.
    """

    def __init__(self, gm):
        super().__init__(gm)
        self.values = {}

    def run_node(self, n):
        out = super().run_node(n)
        self.values[n] = out
        return out


def accum_scales(graph_module, inputs) -> list:
    """Per output: `(S, n)` where S is the accumulated magnitude of each element and
    n is the number of terms accumulated -- or `(None, 0)` if the output is not an
    accumulation.

    WHY THE TOLERANCE NEEDS THIS. A float comparison scaled to the OUTPUT magnitude
    is the wrong ruler for a reduction. For a dot product whose terms cancel, the
    result is near zero while the ERROR is set by the magnitude of the terms that
    were summed. Standard bound: a floating-point sum of n terms differs from the
    exact value by at most n*u*sum|terms| (u the accumulator's unit roundoff), so
    two DIFFERENT valid summation orders differ by at most 2*n*u*sum|terms|.

    MEASURED on `fp16_vecdot` (1024 dot products of length 512): the 6 elements that
    failed the output-scaled tolerance have |want| between 0.0015 and 0.52 against
    an accumulated magnitude of 305-348. numpy's own `sum` failed the same 6, as did
    an explicit pairwise tree -- so no vectorised implementation could pass. With
    the 2*n*u*S term added, both orderings pass 0/1024 while an fp16 ACCUMULATOR --
    the classic error, which this repo has shipped once before -- still fails
    84/1024. The tolerance discriminates; it does not merely loosen.
    """
    rec = _Recorder(graph_module)
    with torch.no_grad():
        raw = rec.run(*inputs)
    outs = (raw,) if isinstance(raw, torch.Tensor) else tuple(raw)

    # map produced tensor -> the node that produced it, by identity
    by_id = {id(v): node for node, v in rec.values.items()
             if isinstance(v, torch.Tensor)}

    scales = []
    for t in outs:
        node = by_id.get(id(t))
        kind = _ACCUM_SCALE.get(str(node.target)) if node is not None else None
        if kind == "reduce_dims":
            src = rec.values[node.all_input_nodes[0]].to(torch.float32).abs()
            dims = list(node.args[1]) if len(node.args) > 1 and node.args[1] else                 list(range(src.dim()))
            dims = [d % src.dim() for d in dims]
            n = 1
            for d in dims:
                n *= src.shape[d]
            # keepdim, then broadcast to the OUTPUT's element count so the harness
            # can index S with the same flat index it uses for the value.
            S = src.sum(dim=dims, keepdim=True).expand(
                *[1 if i in dims else src.shape[i] for i in range(src.dim())])
            scales.append((S.reshape(-1).numpy().copy(), n))
        elif kind == "matmul":
            A = rec.values[node.all_input_nodes[0]].to(torch.float32).abs()
            B = rec.values[node.all_input_nodes[1]].to(torch.float32).abs()
            scales.append(((A @ B).reshape(-1).numpy().copy(), A.shape[-1]))
        else:
            scales.append((None, 0))
    return scales


def _numpy_dtype_key(arr: Any) -> str:
    name = arr.dtype.name
    if name not in DTYPE_C:
        raise ValueError(f"unsupported numpy dtype for harness emission: {name}")
    return name


def _float_literal(value: float) -> str:
    """A C99 hex-float literal (e.g. ``0x1.999ap-4f``) for `value`.

    Bit-exact by construction: `float.hex()` renders the double that exactly
    holds the (float32-or-narrower) golden value, and converting that hex
    string back with a trailing ``f`` rounds to the nearest `float` -- which,
    since the value already fit exactly in float32, is the original value's
    exact bit pattern. Decimal formatting can't make that guarantee; hex can.

    NON-FINITE VALUES HAVE NO HEX FORM AND NEED NAMES. `float("nan").hex()` is
    `'nan'`, so appending `f` produced `nanf` -- which C parses as the FUNCTION
    `nanf`, giving "cannot initialize an array element of type 'const float' with an
    lvalue of type 'float (const char *)'". Same for `inf` -> `inff`.

    That is not hypothetical and it is not only about weird ops: a GOLDEN legitimately
    contains NaN whenever the reference is evaluated outside its domain, and
    `binary_cross_entropy` on a [0.5, 1.5) draw takes `log(1 - x)` with x > 1. The
    harness would not compile, so the batch reported a reference failure with a
    compiler error rather than anything about the kernel.

    `_render_scalar` in `primitives.py` already carries this exact fix for scalar
    ARGUMENTS ("INFINITY AND NAN NEED NAMES, NOT REPR"), and the golden emitter needed
    it too -- the same bug in the same repository, in the other of the two places a
    float becomes C text.
    """
    v = float(value)
    if v != v:
        return "NAN"
    if v == float("inf"):
        return "INFINITY"
    if v == float("-inf"):
        return "(-INFINITY)"
    return f"{v.hex()}f"


def _c_literal(value: Any, dtype_str: str) -> str:
    if dtype_str in ("float32", "float16"):
        return _float_literal(float(value))
    return str(int(value))


def _flat_initializer(arr: Any, dtype_str: str) -> str:
    return ", ".join(_c_literal(v, dtype_str) for v in arr.reshape(-1).tolist())


# Above this many elements a buffer is emitted as a base64 BYTE IMAGE decoded at
# startup, instead of as one C literal per element.
#
# MEASURED, and it is not a micro-optimisation. A hex-float literal costs ~20
# characters per value however wide the value is, so the harness runs 10-12x the
# working set in bytes -- and a T3 task is by definition BIGGER than the 8 MB
# VTCM, which is exactly where DMA and VTCM (the mechanisms this corpus is short
# of) become interesting. Batch 3's `fp16_softmax_stream` came out as a **145 MB
# translation unit**, batch 4's `fp32_silu_stream` as 106 MB, and clang spends
# minutes to tens of minutes parsing those before the simulator sees anything.
#
# The byte image costs 4 characters per 3 bytes, so the saving scales with how
# many characters a literal of that dtype was costing. Measured, same kernels,
# before -> after:
#
#     fp16_softmax_stream    145.4 MB -> 17.4 MB   8.4x
#     fp16_matmul_stream      16.2 MB ->  2.0 MB   8.1x
#     fp32_silu_stream       105.8 MB -> 25.6 MB   4.1x
#     i8_depthwise_conv2d     68.2 MB -> 43.6 MB   1.6x   (int32 golden dominates)
#
# fp16 gains most (a 2-byte value was costing ~20 characters), fp32 about half
# that, and a narrow integer least because its decimal form was already short.
# It is bit-exact by construction: the array's own little-endian bytes are
# stored, not a rendering of them, so no formatting or parsing step can round.
#
# Small buffers keep the literal form deliberately. A 512-element harness that a
# human can read and diff has repeatedly been the thing that identified a wrong
# GOLDEN as opposed to a wrong kernel, and it costs nothing.
B64_THRESHOLD = 4096

_B64_ALPHABET = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                 "0123456789+/")


def _b64_chunks(raw: bytes, per_line=120) -> list:
    """`raw` as base64, split into adjacent C string literals.

    Adjacent literals rather than one enormous one: the standard only guarantees
    65,536 characters in a single string literal, and concatenation is the
    portable way to exceed that.
    """
    import base64
    text = base64.b64encode(raw).decode("ascii")
    return [f'"{text[i:i + per_line]}"' for i in range(0, len(text), per_line)]


def _b64_decoder_c() -> list:
    """A base64 decoder, emitted into the harness.

    Deliberately trivial and table-free: the alphabet is decoded by range tests
    so there is no second table to keep in step with the encoder, and `=` padding
    just stops the output early. This runs once at startup on data the harness
    itself produced, so it is not on any measured path.
    """
    return [
        "/* base64 -> bytes, for the golden buffers emitted as a byte image.",
        " * See oracle.B64_THRESHOLD for why they are not element literals. */",
        "static int forge2_b64_val(char c) {",
        "    if (c >= 'A' && c <= 'Z') return c - 'A';",
        "    if (c >= 'a' && c <= 'z') return c - 'a' + 26;",
        "    if (c >= '0' && c <= '9') return c - '0' + 52;",
        "    if (c == '+') return 62;",
        "    if (c == '/') return 63;",
        "    return -1;                 /* '=' padding, or anything unexpected */",
        "}",
        "static void forge2_b64(const char *src, void *dst, unsigned long nbytes) {",
        "    unsigned char *out = (unsigned char *)dst;",
        "    unsigned long w = 0;",
        "    unsigned acc = 0;",
        "    int have = 0;",
        "    for (const char *p = src; *p; p++) {",
        "        int v = forge2_b64_val(*p);",
        "        if (v < 0) continue;",
        "        acc = (acc << 6) | (unsigned)v;",
        "        have += 6;",
        "        if (have >= 8) {",
        "            have -= 8;",
        "            if (w < nbytes) out[w++] = (unsigned char)((acc >> have) & 0xFF);",
        "        }",
        "    }",
        "    /* Fail loudly rather than run against a half-filled buffer: a short",
        "     * decode would look exactly like a wrong kernel. */",
        "    if (w != nbytes) {",
        '        printf("HVXENV_INCORRECT errors=-1 n=%lu first_bad=-1 (b64 decoded '
        '%lu of %lu bytes)\\n", nbytes, w, nbytes);',
        "        exit(1);",
        "    }",
        "}",
        "",
    ]


def _align_attr() -> str:
    """``__attribute__((aligned(N)))`` for the active target's vector width.

    An HVX ALIGNED vector load/store (``*(HVX_Vector*)ptr``, what ``vmem``
    compiles to) requires the pointer to be aligned to the vector width --
    128 bytes on every target this repo has, but taken from
    `target.current().hvx_bytes` rather than hardcoded so this keeps working
    if that ever changes. A buffer declared without this lands at ordinary
    (4-byte) alignment; a correct hand-vectorised candidate reading it then
    misreads, which is reported as a wall of wrong values starting at index
    0 -- indistinguishable, from the verdict alone, from a broken candidate.
    The harness exists to judge accelerated kernels, so this is the one
    direction it must never fail in.

    MEASURED, not theorised. On the relu graph, same generated harness both
    times: the emitted scalar reference passed, while a correct hand-written
    `Q6_Vsf_vmax_VsfVsf` kernel reported `errors=392 n=512 first_bad=0`. Wrong
    from index 0 is the misalignment signature. Without this the pipeline would
    have concluded "the model cannot write HVX" from an array declaration.
    """
    return f"__attribute__((aligned({_target.current().hvx_bytes})))"


def harness_c(graph, gold: dict) -> str:
    """A complete C++ `main()` that declares `gold`'s inputs/expected outputs
    as initialisers, calls `candidate_kernel`, compares, and prints a
    machine-parseable verdict.

    SELF-CONTAINED BY REQUIREMENT. This file includes only the C++ standard
    headers: it does NOT include `harness_common.h`, and the compile that builds
    it does not put that header's directory on the include path at all (see
    `forge2.verify`). Everything it needs -- the tolerance compares and the HMX
    context enable -- is emitted inline below. The reason is provenance, not
    taste: `hexbench/env/harness/` is R&D material, and a forge v2 translation
    unit that reaches into it has a weaker chain than this pipeline claims. An
    unreachable header cannot leak; a forbidden one can.

    Comparison: exact equality for integer dtypes; for float, an abs+rel
    tolerance AT THE RESULT'S OWN WIDTH -- 1e-4 / 1e-3 for fp32, 4e-3 / 8e-3 for
    fp16, both ~1-2 ULP of their own dtype. Branching on width and not merely on
    float-vs-int is the point: fp32's tolerance is ~1-2 fp16 ULP, so applying it
    to an fp16 result silently makes accumulator precision decide correctness.
    Prints the first mismatching (flattened, across all outputs) index on
    failure. Exit code 0 on match, 1 on mismatch.

    `main()` enables the HMX context first; see the comment at that line for why
    it is here and unconditional.

    The candidate is declared ``extern "C"`` -- this file is compiled as
    C++, and without that the emitted kernel's C linkage symbol won't match
    a mangled C++ declaration.
    """
    in_dtypes = [node.dtype for node in graph.inputs]
    out_dtypes = [_numpy_dtype_key(arr) for arr in gold["outputs"]]
    # Per output: (accumulated magnitude array, term count), or (None, 0). Defaults
    # so a `gold` dict from an older caller still works and produces the identical
    # harness it did before.
    accum = gold.get("accum") or [(None, 0)] * len(out_dtypes)

    lines = [
        "#include <cstdio>",
        "#include <cstdint>",
        "#include <cstdlib>",     # exit(), for the base64 decoder's short-read guard
        "#include <cmath>",
        "",
        "/* ---- everything this harness needs, emitted rather than included ----",
        " * Self-contained on purpose: no header from hexbench/env/harness/ is on",
        " * this compile's include path, so the R&D material cannot reach a forge v2",
        " * translation unit even by accident. See oracle.harness_c's docstring. */",
        "",
        "/* Enable the HMX extension context: SSR.XE (bit 29) + SSR.XA=2 (bits",
        " * 27:25). The standalone runtime enables HVX only, so without this an",
        " * `mxmem` access raises exception 0x18. The program runs privileged. */",
        "static inline void forge2_hmx_enable(void) {",
        "    unsigned m = (1u << 29) | (2u << 25);  /* 0x24000000 */",
        "    unsigned t;",
        '    __asm__ volatile("%0=ssr\\n\\t %0=or(%0,%1)\\n\\t ssr=%0\\n\\t isync\\n\\t"',
        '                     : "=&r"(t) : "r"(m));',
        "}",
        "",
    ]

    # The tolerance compares, one per float width actually present in the outputs.
    # Emitted on demand so an all-integer harness contains no tolerance code at
    # all: "this graph is judged bit-exact" is then visible in the harness itself
    # rather than resting on which branch the comparison loop happened to take.
    if any(dt in ("float32", "float16") for dt in out_dtypes):
        lines += [
            "/* Tolerance compare for FLOATING-POINT results. HVX float arithmetic",
            " * goes through the non-IEEE qfloat path and reductions reorder, so a",
            " * correct vectorised kernel is not bit-identical to the IEEE scalar",
            " * golden. Each dtype is judged at ~1-2 ULP OF ITSELF; integer results",
            " * stay exact and get no tolerance at all. */",
        ]
    if "float32" in out_dtypes:
        lines += [
            "static inline int forge2_close_f32(float g, float e) {",
            "    float d = g - e; if (d < 0) d = -d;",
            "    float ae = e < 0 ? -e : e;",
            "    return d <= 1e-4f + 1e-3f * ae;",
            "}",
        ]
    if "float16" in out_dtypes:
        lines += [
            "static inline int forge2_close_f16(float g, float e) {",
            "    float d = g - e; if (d < 0) d = -d;",
            "    float ae = e < 0 ? -e : e;",
            "    return d <= 4e-3f + 8e-3f * ae;",
            "}",
        ]
    if any(a[0] is not None for a in accum):
        lines += [
            "/* Tolerance for an output that IS AN ACCUMULATION.",
            " *",
            " * Scaling a float tolerance to the OUTPUT magnitude is the wrong ruler",
            " * for a reduction: when the terms cancel the result is near zero while",
            " * the error is set by the magnitude of what was summed. The standard",
            " * bound is |computed - exact| <= n*u*sum|terms| with u the accumulator's",
            " * unit roundoff, so two DIFFERENT valid summation orders differ by at",
            " * most 2*n*u*sum|terms|. `knu` is that 2*n*u, computed once per output;",
            " * `s` is this element's own sum|terms|.",
            " *",
            " * MEASURED on fp16_vecdot: the 6 elements that failed the output-scaled",
            " * tolerance had |want| 0.0015-0.52 against sum|terms| of 305-348, and",
            " * numpy's own sum failed the same 6. With this term both a sequential",
            " * and a pairwise summation pass 0/1024, while an fp16 ACCUMULATOR still",
            " * fails 84/1024 -- it discriminates, it does not merely loosen. */",
            "static inline int forge2_close_acc(float g, float e, float s, float knu,",
            "                                  float atol, float rtol) {",
            "    float d = g - e; if (d < 0) d = -d;",
            "    float ae = e < 0 ? -e : e;",
            "    return d <= atol + rtol * ae + knu * s;",
            "}",
        ]
    lines.append("")

    params = [f"const {DTYPE_C[dt]}* in{i}" for i, dt in enumerate(in_dtypes)]
    params += [f"{DTYPE_C[dt]}* out{i}" for i, dt in enumerate(out_dtypes)]
    lines.append(f'extern "C" void candidate_kernel({", ".join(params)});')
    lines.append("")

    # Every buffer the candidate touches is aligned to the vector width; see
    # _align_attr() for why this is load-bearing rather than cosmetic.
    attr = _align_attr()

    # Buffers, either as element literals or as a base64 byte image (see
    # B64_THRESHOLD). `decode_calls` collects the startup work for the image
    # form; a harness with no large buffer emits neither the decoder nor a call,
    # so small harnesses are byte-for-byte what they were before.
    decode_calls = []
    big = [arr for arr in list(gold["inputs"]) + list(gold["outputs"])
           + [a[0] for a in accum if a[0] is not None]
           if arr.size >= B64_THRESHOLD]
    if big:
        lines.extend(_b64_decoder_c())

    for i, (arr, dt) in enumerate(zip(gold["inputs"], in_dtypes)):
        if arr.size >= B64_THRESHOLD:
            # NOT const: it is filled at startup. The candidate still sees it as
            # `const T*`, so nothing about the kernel's contract changes.
            lines.append(f"static {DTYPE_C[dt]} golden_in_{i}[{arr.size}] {attr};")
            lines.append(f"static const char b64_in_{i}[] =")
            lines.extend(f"    {c}" for c in _b64_chunks(arr.tobytes()))
            lines.append("    ;")
            decode_calls.append(
                f"    forge2_b64(b64_in_{i}, golden_in_{i}, sizeof(golden_in_{i}));")
        else:
            lines.append(
                f"static const {DTYPE_C[dt]} golden_in_{i}[{arr.size}] {attr} = "
                f"{{ {_flat_initializer(arr, dt)} }};"
            )
    lines.append("")

    for i, (arr, dt) in enumerate(zip(gold["outputs"], out_dtypes)):
        if arr.size >= B64_THRESHOLD:
            lines.append(f"static {DTYPE_C[dt]} golden_out_{i}[{arr.size}] {attr};")
            lines.append(f"static const char b64_out_{i}[] =")
            lines.extend(f"    {c}" for c in _b64_chunks(arr.tobytes()))
            lines.append("    ;")
            decode_calls.append(
                f"    forge2_b64(b64_out_{i}, golden_out_{i}, sizeof(golden_out_{i}));")
        else:
            lines.append(
                f"static const {DTYPE_C[dt]} golden_out_{i}[{arr.size}] {attr} = "
                f"{{ {_flat_initializer(arr, dt)} }};"
            )
        lines.append(f"static {DTYPE_C[dt]} out_{i}[{arr.size}] {attr};")
        S, n = accum[i]
        if S is not None:
            # float32 always: the scale is a magnitude bound, not a value of the
            # output's dtype, and it is compared against in float.
            if S.size >= B64_THRESHOLD:
                lines.append(f"static float accum_s_{i}[{S.size}] {attr};")
                lines.append(f"static const char b64_acc_{i}[] =")
                lines.extend(f"    {c}" for c in _b64_chunks(S.tobytes()))
                lines.append("    ;")
                decode_calls.append(
                    f"    forge2_b64(b64_acc_{i}, accum_s_{i}, sizeof(accum_s_{i}));")
            else:
                lines.append(
                    f"static const float accum_s_{i}[{S.size}] {attr} = "
                    f"{{ {', '.join(_float_literal(v) for v in S.reshape(-1).tolist())} }};")
    lines.append("")

    lines.append("int main() {")
    # The HMX context, enabled here because the harness is the only place that
    # CAN enable it and the only place that should. Without it every HMX
    # candidate faults with exception 0x18 on its first `mxmem`, which reads as
    # the author's bug. Both batch-3 HMX authors worked around it inside their
    # kernels; that is the workaround this removes. Same class as the missing
    # `--mhmx 2` in `verify.py`: the sim-side half of enabling HMX was fixed, the
    # runtime half was not.
    #
    # Unconditional, not gated on the graph looking like a matmul. Entitlement
    # is not knowable here (this function sees the graph and the golden, not the
    # mechanism plan), and a candidate may legitimately reach for HMX on a graph
    # that was not entitled to it. Writing SSR is harmless for a kernel that
    # never issues an HMX instruction -- the standalone image runs privileged and
    # this only ORs bits into a register.
    lines.append("    forge2_hmx_enable();")
    # Decode before anything reads a golden buffer. Ordering is load-bearing: a
    # kernel run against a still-zero input would fail as if it were wrong.
    lines.extend(decode_calls)
    lines.append("    int total_errors = 0;")
    lines.append("    int total_n = 0;")
    lines.append("    int first_bad = -1;")
    # Accumulators for the SHAPE of the bad set. See the print at the end for what
    # each pattern means and which real bug it named.
    lines.append("    int last_bad = -1, bad_stride = -1, run_len = 0;")
    lines.append("    int r32 = -1, r64 = -1, r128 = -1;")
    lines.append("    int same32 = 1, same64 = 1, same128 = 1, uniform = 1;")
    call_args = [f"golden_in_{i}" for i in range(len(in_dtypes))]
    call_args += [f"out_{i}" for i in range(len(out_dtypes))]
    lines.append(f"    candidate_kernel({', '.join(call_args)});")
    lines.append("")

    for i, dt in enumerate(out_dtypes):
        n = gold["outputs"][i].size
        lines.append(f"    for (int i = 0; i < {n}; i++) {{")
        S, n_terms = accum[i]
        if dt in ("float16", "float32") and S is not None:
            # An accumulation: add the term the reduction's error is really bounded
            # by. u = 2^-24 is the fp32 accumulator's unit roundoff (every emitted
            # reference accumulates at least that wide -- see accum_ctype).
            atol, rtol = (("4e-3f", "8e-3f") if dt == "float16"
                          else ("1e-4f", "1e-3f"))
            knu = 2.0 * n_terms * (2.0 ** -24)
            cond = (f"!forge2_close_acc((float)out_{i}[i], "
                    f"(float)golden_out_{i}[i], accum_s_{i}[i], "
                    f"{_float_literal(knu)}, {atol}, {rtol})")
        elif dt == "float16":
            # The fp16 ruler for an fp16 result, deliberately. Previously this
            # branch was float-vs-int and fp16 outputs went through the fp32
            # tolerance, i.e. ~1-2 fp16 ULP -- tight enough that a second ULP of
            # accumulator drift decided correctness.
            cond = f"!forge2_close_f16((float)out_{i}[i], (float)golden_out_{i}[i])"
        elif dt == "float32":
            cond = f"!forge2_close_f32((float)out_{i}[i], (float)golden_out_{i}[i])"
        else:
            cond = f"out_{i}[i] != golden_out_{i}[i]"
        lines.append(f"        if ({cond}) {{")
        lines.append("            int bi = total_n + i;")
        lines.append("            if (first_bad < 0) {")
        lines.append("                first_bad = bi;")
        lines.append("                r32 = bi & 31; r64 = bi & 63; r128 = bi & 127;")
        lines.append("                run_len = 1;")
        lines.append("            } else {")
        lines.append("                int gap = bi - last_bad;")
        lines.append("                if (bad_stride < 0) bad_stride = gap;")
        lines.append("                else if (gap != bad_stride) uniform = 0;")
        lines.append("                if (gap == 1 && run_len == bi - first_bad) "
                     "run_len++;")
        lines.append("                if ((bi & 31)  != r32)  same32 = 0;")
        lines.append("                if ((bi & 63)  != r64)  same64 = 0;")
        lines.append("                if ((bi & 127) != r128) same128 = 0;")
        lines.append("            }")
        lines.append("            last_bad = bi;")
        lines.append("            total_errors++;")
        lines.append("        }")
        lines.append("    }")
        lines.append(f"    total_n += {n};")
    lines.append("")

    lines.append("    if (total_errors == 0) {")
    lines.append('        printf("HVXENV_CORRECT errors=0 n=%d\\n", total_n);')
    lines.append("        return 0;")
    lines.append("    } else {")
    # THE SHAPE OF THE BAD SET, NOT JUST ITS SIZE.
    #
    # `errors/n/first_bad` alone says a kernel is wrong; it does not say why, and
    # every numeric bug in this corpus was actually diagnosed from WHICH elements
    # were wrong. Measured examples:
    #
    #   first_bad == the lane count       only the first vector was computed
    #                                     (`N = 64` floats, 32 lanes -- looked like
    #                                     sixteen broken math functions)
    #   bad_stride == 2                   a widening deal read as a concatenation
    #   all bad share a residue mod 64    half-vector pack order
    #   one contiguous run to the end     the tail was never written
    #   ~18% scattered                    a rounding-mode bug (only ties are wrong)
    #
    # Costs one extra comparison per element in a loop that already does a
    # tolerance compare, and it is the only diagnostic an API-driven author can
    # act on -- it cannot read the disassembly the way a local session can.
    lines.append("        /* the SHAPE of the bad set: what names the bug.")
    lines.append("         * mod32/mod64/mod128 are the shared residue when EVERY")
    lines.append("         * bad index has one, and -1 when they do not. */")
    lines.append("        printf(\"HVXENV_INCORRECT errors=%d n=%d first_bad=%d "
                 "last_bad=%d bad_stride=%d uniform_stride=%d run_len=%d "
                 "mod32=%d mod64=%d mod128=%d\\n\",")
    lines.append("               total_errors, total_n, first_bad, last_bad,")
    lines.append("               bad_stride, (bad_stride > 0 ? uniform : 0),")
    lines.append("               run_len,")
    lines.append("               (same32 ? r32 : -1), (same64 ? r64 : -1),")
    lines.append("               (same128 ? r128 : -1));")
    lines.append("        return 1;")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)
