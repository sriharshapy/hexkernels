"""The kernel batch: op, size, and the mechanisms the size implies.

Five at a time, deliberately. A batch that is five wide and broken is worth less
than five that run, and every batch so far has surfaced at least one integration
defect that only appears when a kernel is actually compiled and executed --
buffer alignment in the generated harness, mechanism detection reading the linked
ELF instead of the candidate. Widening the batch would have hidden those behind
aggregate counts.

CHOOSING THE OPS
----------------
Drawn from the harvested registry (`run_artifacts/forge2/ops.jsonl`), filtered by
`mechanism.mechanism_eligible` -- the op must do arithmetic and move data, so a
kernel for it can say something about the accelerator. `view`, `expand` and the
rest of the metadata ops emit perfectly and teach nothing.

CHOOSING THE SIZES
------------------
The size is not decoration and not arbitrary: it is what entitles a kernel to a
mechanism (`mechanism.plan_for`). Each entry below records the tier it is meant
to land in, and `test_kernels.py` asserts it still does -- a shape ladder that
silently stops straddling the thresholds is a defect this project has shipped
twice already, in a different module, both times unnoticed because every other
test still passed.

Batch 1 stays inside T1/T2 (L1-to-VTCM). Those exercise HVX and l2fetch, which is
what the emitters currently support; T3 streaming needs DMA double-buffering,
which is a later batch and a different rewrite.
"""
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class KernelSpec:
    """One benchmark task before it has been traced."""

    name: str
    module: torch.nn.Module
    args: tuple
    dtype_bytes: int
    expect_tier: str
    note: str
    # Mechanisms the size is expected to justify. Asserted, not assumed --
    # if the derivation stops agreeing with the intent, one of the two is wrong.
    expect_mechanisms: frozenset = field(default_factory=frozenset)
    # The part `expect_tier` and `expect_mechanisms` are claims ABOUT. A tier is a
    # comparison against one memory hierarchy, so it is meaningless without saying
    # which: batch 2's `fp32_relu_stream` is 8 MB, which is T2 on v75 (VTCM 8 MB)
    # and T3 on v68 (VTCM 4 MB). Both are correct readings of the same shape. This
    # field is what lets the assertion be checked against the hierarchy the kernel
    # was built and verified on, rather than against whatever `HEXBENCH_TARGET`
    # happens to be set to when the test runs.
    target: str = "v75"


class Relu(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x)


class AddRelu(torch.nn.Module):
    def forward(self, a, b):
        return torch.relu(a + b)


class Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, 1)


class MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class Int8Elementwise(torch.nn.Module):
    """Integer requant chain in plain tensor ops.

    Written without qtensor deliberately: `torch.quantize_per_tensor` does not
    survive `torch.export`, and the int8 path is the half of the benchmark where
    the accelerator's integer units matter most.
    """

    def forward(self, a, b):
        s = a.to(torch.int32) + b.to(torch.int32)
        return torch.clamp(s * 3 >> 2, -128, 127).to(torch.int8)


def batch1():
    """Five kernels spanning elementwise, reduction, contraction and integer."""
    return (
        KernelSpec(
            name="relu_fp32", module=Relu(), args=(torch.randn(64, 1024),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="simplest possible end-to-end proof; one parallel nest, unit "
                 "stride, so the whole kernel is one vector loop",
        ),
        KernelSpec(
            name="add_relu_fp32", module=AddRelu(),
            args=(torch.randn(64, 1024), torch.randn(64, 1024)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="two inputs; tests that operand order survives the trace and "
                 "that both loads are vectorised",
        ),
        KernelSpec(
            name="softmax_fp32", module=Softmax(), args=(torch.randn(256, 256),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="reduction plus broadcast: amax/sub/exp/sum/div. The two "
                 "reductions do NOT vectorise as plain vector loops and the "
                 "three elementwise steps do -- the case where the schedule "
                 "annotation says something the scalar reference cannot",
        ),
        KernelSpec(
            name="matmul_fp32", module=MatMul(),
            args=(torch.randn(256, 256), torch.randn(256, 256)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="contraction. The loop shape is HMX's, but the dtype is not: "
                 "HMX is an int8/fp16 tile engine and there is no fp32 tile "
                 "multiply, so this task is deliberately HVX-only. An fp16 or "
                 "int8 matmul in a later batch is where HMX becomes reachable",
        ),
        KernelSpec(
            name="int8_elementwise", module=Int8Elementwise(),
            args=(torch.randint(-128, 127, (128, 1024), dtype=torch.int8),
                  torch.randint(-128, 127, (128, 1024), dtype=torch.int8)),
            dtype_bytes=1, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="integer path: widening casts, a scalar multiply, an "
                 "arithmetic shift and a saturating clamp back to int8",
        ),
    )


class Fp16MatMul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class Int8MatMul(torch.nn.Module):
    """int8 operands accumulated in int32 -- how a quantised GEMM is expressed
    without qtensor, which does not survive `torch.export`."""

    def forward(self, a, b):
        return a.to(torch.int32) @ b.to(torch.int32)


class Int8Add(torch.nn.Module):
    def forward(self, a, b):
        return torch.clamp(a.to(torch.int32) + b.to(torch.int32), -128, 127).to(torch.int8)


def batch2():
    """The mechanisms batch 1 could not reach.

    Batch 1 landed entirely at T1 and entirely on HVX: every size fit L2, so DMA
    and VTCM were never justified, and every dtype was fp32, so HMX was never
    satisfiable. Both are deliberate here -- the dtype opens HMX, the size opens
    DMA/VTCM. Those are also the mechanisms this project has historically failed
    to elicit, so they are the ones worth building tasks for.
    """
    return (
        KernelSpec(
            name="fp16_matmul", module=Fp16MatMul(),
            args=(torch.randn(256, 256, dtype=torch.float16),
                  torch.randn(256, 256, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "hmx", "l2fetch"}),
            note="the honest HMX case: contraction shape AND a dtype the tile "
                 "engine can take. Batch 1's matmul had the shape and not the dtype",
        ),
        KernelSpec(
            name="i8_matmul", module=Int8MatMul(),
            args=(torch.randint(-8, 8, (256, 256), dtype=torch.int8),
                  torch.randint(-8, 8, (256, 256), dtype=torch.int8)),
            dtype_bytes=1, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "hmx", "l2fetch"}),
            note="int8 GEMM with int32 accumulation, the quantised-inference "
                 "shape; HMX's native datapath. (T1, not T0 as first written: "
                 "three 64K-element int8 tensors are 192 KB, well past L1D. The "
                 "expect_tier field caught that arithmetic slip.)",
        ),
        KernelSpec(
            name="fp32_relu_stream", module=Relu(),
            args=(torch.randn(1024, 1024),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="8 MB working set: past L2, so the data does not stay resident "
                 "and staging is what buys the bandwidth. Same op as batch 1's "
                 "relu -- ONLY the size differs, which is the whole claim",
        ),
        KernelSpec(
            name="i8_add_stream", module=Int8Add(),
            args=(torch.randint(-100, 100, (2048, 2048), dtype=torch.int8),
                  torch.randint(-100, 100, (2048, 2048), dtype=torch.int8)),
            dtype_bytes=1, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="two streams in, one out: 12 MB, past VTCM as well as L2, so it "
                 "must be streamed in tiles -- the double-buffering case",
        ),
        KernelSpec(
            name="fp32_softmax_stream", module=Softmax(),
            args=(torch.randn(1024, 1024),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="row reduction at a staged size: the reduction must be tiled "
                 "along the row, which is where the schedule annotation and the "
                 "memory plan have to agree",
        ),
    )


class Fp16Conv2d(torch.nn.Module):
    """Dense 3x3 convolution, groups=1 -- a real channel contraction.

    Written with an explicit weight argument rather than an `nn.Conv2d` module so
    the weight is a graph PLACEHOLDER and therefore part of the working set. A
    module parameter would be lifted as a constant and the size arithmetic that
    licenses the mechanism would be quietly wrong.
    """

    def forward(self, x, w):
        return torch.nn.functional.conv2d(x, w, None, 1, 1)


class Fp16AttentionScores(torch.nn.Module):
    """`softmax(Q @ K^T / sqrt(d))` -- the attention score block.

    K arrives already transposed, as `(d, N)`. That is not a simplification: a
    transpose is a metadata op with no arithmetic (`mechanism.NO_ARITHMETIC`), so
    tracing `k.transpose(0, 1)` would add a node the emitter has no kernel for and
    change nothing about the hardware question. Real attention kernels are handed
    a K^T view for the same reason.

    0.125 is 1/sqrt(64), exact in binary, so the scale contributes no rounding of
    its own to an fp16 comparison.
    """

    def forward(self, q, kt):
        return torch.softmax(q @ kt * 0.125, 1)


class I8DepthwiseConv2d(torch.nn.Module):
    """3x3 depthwise convolution: one filter per channel, groups == channels (512).

    int8 in, accumulated in int32, exactly as a quantised depthwise layer runs --
    `aten.convolution` has no int8 overload, and `torch.quantize_per_tensor` does
    not survive `torch.export`, so the widening casts ARE the honest spelling.
    """

    def forward(self, x, w):
        return torch.nn.functional.conv2d(
            x.to(torch.int32), w.to(torch.int32), None, 1, 1, 1, 512)


class Fp16Softmax(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, 1)


def batch3():
    """Push the three scarce mechanisms: DMA, VTCM, HMX.

    Batches 1-2 finished at dma 3/10, vtcm 4/10, hmx 1/10 -- the mechanisms this
    project has historically failed to elicit at all. Every kernel here is sized
    past L2, so DMA and VTCM are entitled in all five rather than in three of ten,
    and three of them are contractions in a dtype the tile engine accepts.

    A convolution appears for the first time. Its emitter has existed since the
    frontend was built, but `schedule_of` had no rule for it, so the pipeline
    would have raised at the annotate stage -- the two halves were never run
    together. The rule added for it is also what forced the affine `Term`
    representation, since `oh*stride + kh*dilation` is two loops indexing one
    operand axis and the old table could not say that.

    SIZES SIT JUST PAST THEIR THRESHOLD, ON PURPOSE
    -----------------------------------------------
    Each shape clears its tier floor by 1.07x-1.53x rather than by 6x. The
    entitlement is a binary comparison against a probed constant, so overshooting
    buys nothing -- and it costs twice over, because `oracle` inlines the whole
    working set into the harness as decimal literals. A first cut of this batch
    used a 1024-cubed fp16 matmul (6.3 MB, still T2), which produced a **74 MB
    translation unit** that `hexagon-clang++` was still chewing on after 12
    minutes; the same tier at 1.44 MB is a 6 MB harness. Harness bytes run about
    10x the working set for float dtypes, which is the practical ceiling on T3
    tasks and is worth knowing before choosing a shape.

    Shapes are also non-square where they can be (1024x128 @ 128x512, not 512
    cubed), so a candidate cannot pass by accident on a symmetry.
    """
    return (
        KernelSpec(
            name="fp16_matmul_stream", module=Fp16MatMul(),
            args=(torch.randn(1024, 128, dtype=torch.float16),
                  torch.randn(128, 512, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "hmx", "l2fetch", "dma", "vtcm"}),
            note="all five mechanisms in one task, which no earlier kernel "
                 "reached: the contraction shape and an fp16 dtype open HMX, and "
                 "1.44 MB is past L2 so the tiles must be staged. Batch 2's "
                 "fp16_matmul is the same op one size down, at T1, where staging "
                 "was not justified -- so the pair isolates the size variable",
        ),
        KernelSpec(
            name="fp16_conv2d_3x3", module=Fp16Conv2d(),
            args=(torch.randn(1, 64, 64, 64, dtype=torch.float16),
                  torch.randn(64, 64, 3, 3, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "hmx", "l2fetch", "dma", "vtcm"}),
            note="the first convolution, and a second independent route to HMX: "
                 "groups=1 contracts over 64 input channels, which is a tile "
                 "matmul with a sliding window on top. Also the first task whose "
                 "input map has a coefficient (oh*1 + kh*1), so it is the one "
                 "that exercises the affine representation end to end",
        ),
        KernelSpec(
            name="fp16_attention_scores", module=Fp16AttentionScores(),
            args=(torch.randn(768, 64, dtype=torch.float16),
                  torch.randn(64, 768, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "hmx", "l2fetch", "dma", "vtcm"}),
            note="a contraction FEEDING a reduction: mm, scale, then the "
                 "amax/sub/exp/sum/div chain. The intermediate scores matrix is "
                 "16x larger than either input, so where it lives is the whole "
                 "problem -- the case where the mechanism plan and the schedule "
                 "annotation have to be read together",
        ),
        KernelSpec(
            name="i8_depthwise_conv2d_stream", module=I8DepthwiseConv2d(),
            args=(torch.randint(-8, 8, (1, 512, 112, 112), dtype=torch.int8),
                  torch.randint(-4, 4, (512, 1, 3, 3), dtype=torch.int8)),
            dtype_bytes=1, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="grouped convolution at 12.8 MB -- past VTCM, so double-buffered. "
                 "HMX is deliberately NOT expected: groups == channels (512) means "
                 "nothing is contracted across channels, so there is no tile "
                 "matmul to map it onto. The honest non-grant, and the assertion "
                 "that `mechanism._contracts` reads `groups` rather than the op name",
        ),
        KernelSpec(
            name="fp16_softmax_stream", module=Fp16Softmax(),
            args=(torch.randn(1792, 1792, dtype=torch.float16),),
            dtype_bytes=2, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="12.8 MB row reduction: past VTCM, so a row cannot even be held "
                 "whole and the reduction itself has to be tiled across transfers. "
                 "Batch 2's fp32 softmax at T2 could stage a row and reduce it in "
                 "place; this one cannot, which is the harder half of the pair",
        ),
    )


class Fp16Gelu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.gelu(x)


class Fp32Silu(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.silu(x)


class Fp16HardSwish(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.hardswish(x)


class I8Threshold(torch.nn.Module):
    """A COMPARISON, so the result is a bool tensor -- the point of this kernel."""

    def forward(self, a, b):
        return a > b


class Fp16VecDot(torch.nn.Module):
    """Row-wise dot product, written as its own decomposition.

    `torch.linalg.vecdot(a, b)` is the op this kernel was selected from, and the
    FX path handles it -- `run_decompositions` opens it into exactly the mul and
    sum below. The LINALG path does not: torch-mlir's frontend fails to legalize
    `torch.aten.linalg_vecdot` ("explicitly marked illegal"), and both IRs are
    required pipeline inputs, not one-of-two. Writing the decomposition here
    keeps the primitives identical -- `aten.mul.Tensor` and `aten.sum.dim_IntList`,
    both harvested and both selected -- while giving torch-mlir ops it can lower.
    """

    def forward(self, a, b):
        return (a * b).sum(dim=-1)


def batch4():
    """THE FIRST BATCH THE HARVEST CHOSE.

    Batches 1-3 picked ops by hand from the ~15 the emitter table happened to
    support. Every kernel here was instead selected from
    `forge2.coverage`'s measured `covered` list -- ops proven expressible by
    tracing them and emitting C, not by anyone's recollection of what works. That
    closes the gap `CLAUDE.md` has recorded as open since batch 3: the harvest
    now drives selection, and not only provenance.

    The pipeline change that made this batch possible, and its measurement:
    coverage went from **24 eligible ops to 99** (2.1% -> 8.5% of 1,170). It came
    almost entirely from two things the coverage scan RANKED rather than guessed:

      * `bool` had no C type. 63 eligible ops decompose through a boolean
        intermediate -- every comparison, `where`, the logical family, the
        `isfinite`/`isnan` predicates -- and all 63 failed with `KeyError: 'bool'`
        in `DTYPE_C`, which reads as "no emitter" while being a missing dtype.
      * ~40 elementwise rows and 5 reduction rows, added in the order the scan
        ranked them by ops-unblocked.

    WHY THESE FIVE
    --------------
    Four of the five are ops that could not be traced at all a day ago, and each
    exercises a different new part of the pipeline:

      gelu / silu / hardswish  three activations that decompose into DIFFERENT
        primitive chains (erf; sigmoid; clamp+mul), so they test three separate
        groups of new rows rather than three sizes of the same one.
      i8_threshold_mask        the first BOOL-OUTPUT kernel in the corpus. It
        writes one byte per element from a comparison, which is the new dtype
        carried end to end -- traced, emitted, golden-initialised and compared.
        Nothing else here would catch a bool that works in C but not in the
        harness.
      fp16_vecdot              a row-wise dot product: mul then a reduction over
        the last axis. The reduction rows are new, and a per-row horizontal
        reduce is a different HVX problem from a matmul's accumulation.

    Sizes follow batch 3's rule -- just past the tier floor, non-square where
    possible, because the harness inlines the whole working set as literals and
    runs about 10-12x the working set in bytes.
    """
    return (
        KernelSpec(
            name="fp16_gelu", module=Fp16Gelu(),
            args=(torch.randn(768, 1024, dtype=torch.float16),),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="1.5 MB elementwise past L2, so the tiles must be staged. gelu "
                 "decomposes through erf, which has no HVX instruction -- the "
                 "interesting question is what a candidate does about a "
                 "transcendental the datapath does not have: a polynomial "
                 "approximation inside the fp16 tolerance, or scalar fallback",
        ),
        KernelSpec(
            name="fp32_silu_stream", module=Fp32Silu(),
            args=(torch.randn(1536, 1536),),
            dtype_bytes=4, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="9.4 MB, past VTCM: x * sigmoid(x) is one pass with zero reuse, "
                 "so it is a pure streaming problem and the arithmetic is the easy "
                 "half. Pairs with batch 2's fp32_relu_stream (same shape class, "
                 "T2) to isolate what changes when the working set stops fitting",
        ),
        KernelSpec(
            name="fp16_hardswish", module=Fp16HardSwish(),
            args=(torch.randn(48, 64, dtype=torch.float16),),
            dtype_bytes=2, expect_tier="T0",
            expect_mechanisms=frozenset({"hvx"}),
            note="12 KB, inside L1D: the whole task is HVX arithmetic with no "
                 "memory problem at all, which is the control for the two "
                 "streaming kernels above. hardswish is clamp+mul+div -- all "
                 "integer-friendly HVX ops, no transcendental",
        ),
        KernelSpec(
            name="i8_threshold_mask", module=I8Threshold(),
            args=(torch.randint(-100, 100, (384, 768), dtype=torch.int8),
                  torch.randint(-100, 100, (384, 768), dtype=torch.int8)),
            dtype_bytes=1, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the first bool-output task: 288 KB in each, 288 KB of 0/1 bytes "
                 "out -- 864 KB total, inside L2 so no staging is entitled. "
                 "HVX compares produce a PREDICATE register, not a byte vector, so "
                 "a candidate has to materialise the mask (vmux against 1/0) -- a "
                 "different problem from every arithmetic kernel so far",
        ),
        KernelSpec(
            name="fp16_vecdot", module=Fp16VecDot(),
            args=(torch.randn(1024, 512, dtype=torch.float16),
                  torch.randn(1024, 512, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="1024 independent 512-long dot products, 2 MB in: a reduction "
                 "whose reduced axis is the contiguous one, so the HVX work is a "
                 "horizontal reduce per row rather than a vertical accumulate. "
                 "HMX is not expected -- there is no shared contraction to tile, "
                 "each row reduces against its own row",
        ),
    )


class Fp16Atan2(torch.nn.Module):
    def forward(self, a, b):
        return torch.atan2(a, b)


class Fp32LogAddExp(torch.nn.Module):
    def forward(self, a, b):
        return torch.logaddexp(a, b)


class Fp32AmaxAll(torch.nn.Module):
    def forward(self, x):
        return torch.amax(x)


class Fp32RowNorm(torch.nn.Module):
    def forward(self, x):
        return torch.linalg.vector_norm(x, dim=-1)


class Fp32MeanAll(torch.nn.Module):
    def forward(self, x):
        return x.mean()


def batch5():
    """Transcendentals and reductions, and THE SIZES ARE CHOSEN BY THE ABLATION.

    Every op here comes from `forge2.coverage`'s measured `covered` list, and all
    five were unexpressible before this batch's pipeline work (coverage 99 -> 171).

    SIZING RULE, applied for the first time from measurement rather than from the
    tier floor alone. `run_artifacts/forge2/ABLATION.md`: DMA/VTCM staging and
    l2fetch both pay off only above **~3 vector operations per element** -- below
    that the transfer dominates either way and staging is pure overhead. So the
    size is chosen to make the mechanisms the plan grants the ones that actually
    help:

      high intensity (a polynomial, ~15 ops/element)  -> size PAST L2, so dma+vtcm
                                                         are granted AND earned
      low intensity (1-2 ops/element)                 -> size INSIDE L2, so the
                                                         plan grants hvx (+l2fetch)
                                                         and never claims staging
                                                         it cannot justify

    That is also why every shape here is small: the largest working set is 1.5 MB.
    Batch 3 sized a softmax at 12.8 MB for no reason beyond clearing the T3 floor,
    which cost a 145 MB harness and hours of wall clock and bought nothing the
    mechanism story needed.
    """
    return (
        KernelSpec(
            name="fp16_atan2", module=Fp16Atan2(),
            args=(torch.randn(512, 512, dtype=torch.float16),
                  torch.randn(512, 512, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="a TWO-ARGUMENT transcendental, which no earlier kernel has: the "
                 "result depends on the signs of both operands, so a candidate needs "
                 "a core approximation for |y/x| plus quadrant reconstruction. High "
                 "intensity, so 1.5 MB past L2 makes the staging grant one the "
                 "ablation says pays",
        ),
        KernelSpec(
            name="fp32_logaddexp", module=Fp32LogAddExp(),
            args=(torch.randn(384, 384), torch.randn(384, 384)),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="log(exp(a)+exp(b)) computed the stable way, max + log1p(exp(-|d|)): "
                 "15 primitives, the longest chain in any batch, and it needs BOTH "
                 "an exp and a log approximation. High intensity, so 1.77 MB past L2 "
                 "makes the staging grant one the ablation says is earned. Replaced "
                 "erfc, which torch-mlir refuses to legalize (screened, not "
                 "discovered after building a reference)",
        ),
        KernelSpec(
            name="fp32_amax_all", module=Fp32AmaxAll(),
            args=(torch.randn(320, 320),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="whole-tensor MAX to one scalar. Same reduce-everything shape as "
                 "mean_all but with no accumulator width question and no divide, so "
                 "the pair isolates what the reduction operator itself costs. "
                 "Replaced hypot, which torch-mlir refuses to legalize",
        ),
        KernelSpec(
            name="fp32_row_norm", module=Fp32RowNorm(),
            args=(torch.randn(512, 384),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="row-wise 2-norm: multiply-accumulate then one sqrt per row. "
                 "Intensity 2, BELOW the measured staging threshold, so it is sized "
                 "inside L2 and the plan grants no dma -- the honest grant for this "
                 "arithmetic. Pairs with fp16_vecdot: same reduction shape, but the "
                 "terms are squares so nothing cancels and the tolerance question "
                 "vecdot raised does not arise",
        ),
        KernelSpec(
            name="fp32_mean_all", module=Fp32MeanAll(),
            args=(torch.randn(48, 64),),
            dtype_bytes=4, expect_tier="T0",
            expect_mechanisms=frozenset({"hvx"}),
            note="a whole-tensor reduction to ONE scalar -- every axis reduced, "
                 "which is the case that was blocked behind reading a dim argument "
                 "that .default overloads do not have. 12 KB, inside L1D: the entire "
                 "task is a horizontal reduce and there is no memory problem, so hvx "
                 "is the only honest grant",
        ),
    )


class Fp32FlipAdd(torch.nn.Module):
    def forward(self, a, b):
        return torch.flip(a, [1]) + b


class Fp32WhereSelect(torch.nn.Module):
    def forward(self, a, b):
        return torch.where(a > 0, a, b)


class Fp32DiagScale(torch.nn.Module):
    def forward(self, x):
        return torch.diagonal(x) * 2.0


class Fp32Trace(torch.nn.Module):
    def forward(self, x):
        return torch.diagonal(x).sum()


class I32IndexSelect(torch.nn.Module):
    def forward(self, x, idx):
        return torch.index_select(x, 0, idx)


def batch6():
    """Addressing: kernels that read their operand somewhere other than in order.

    THE PROVENANCE GATE REWROTE THIS BATCH, and that is worth recording. The first
    version was permute / slice / expand / arange, and **four of the five kernels
    were DESTROYED** at the provenance stage: those ops are harvested but marked
    "layout/metadata only, no arithmetic", so they do not pass `mechanism_eligible`,
    and the rule requires EVERY primitive in the graph to. A transpose before an add
    is a perfectly ordinary composition and the corpus cannot contain it. That is a
    real tension between the rule as written and what the pipeline can express --
    recorded in RESUME.md as a decision for the user rather than quietly relaxed,
    because loosening a provenance rule is not a change to make while nobody is
    watching.

    The ops below are the ones in this family that ARE eligible, so the batch keeps
    the theme -- non-sequential addressing -- without touching the rule.

    SIZES: all five are 1-3 operations per element, below the ~3-op staging
    break-even from `ABLATION.md`, so all five are sized inside L2 and granted hvx
    (+l2fetch) only. `i32_index_select` is the interesting exception in principle --
    a gather MUST run from VTCM on this hardware (HVX PRM 3.3) -- but its size does
    not reach the staging tier, so the plan does not grant vtcm and the honest
    accelerated form is a scalar-address gather. Recorded as the reason a
    capacity-derived grant and an ISA REQUIREMENT are different things.
    """
    return (
        KernelSpec(
            name="fp32_flip_add", module=Fp32FlipAdd(),
            args=(torch.randn(256, 320), torch.randn(256, 320)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="a REVERSED read along the vectorised axis: input index extent-1-k, "
                 "a unit-stride walk in the negative direction. HVX has no reversing "
                 "load, so the candidate needs an in-register reverse -- and the "
                 "schedule's stride for that loop is -1, which is why "
                 "vectorizable_loop has to consider sign and not just magnitude",
        ),
        KernelSpec(
            name="fp32_where_select", module=Fp32WhereSelect(),
            args=(torch.randn(256, 320), torch.randn(256, 320)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="a three-operand select driven by a comparison. Pairs with batch 4's "
                 "i8_threshold_mask, which MATERIALISED a predicate as bytes; this one "
                 "CONSUMES a predicate directly in vmux and never stores it, so the "
                 "pair shows both halves of how HVX handles a mask",
        ),
        KernelSpec(
            name="fp32_diag_scale", module=Fp32DiagScale(),
            args=(torch.randn(480, 480),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the main diagonal, scaled: one loop indexing BOTH axes of the "
                 "operand, so consecutive outputs are 2052 bytes apart. Rank-reducing, "
                 "which the remap family cannot express -- and not vectorisable as a "
                 "contiguous load at all, which is the honest answer the schedule "
                 "should give. 480x480 keeps it inside L2, so no staging is granted "
                 "for a 1-op-per-element kernel",
        ),
        KernelSpec(
            name="fp32_trace", module=Fp32Trace(),
            args=(torch.randn(480, 480),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the same diagonal walk feeding a reduction to one scalar. Pairs with "
                 "diag_scale to separate the cost of the strided READ from the cost of "
                 "the reduction that follows it",
        ),
        KernelSpec(
            name="i32_index_select", module=I32IndexSelect(),
            args=(torch.randn(384, 256),
                  torch.randint(0, 384, (192,), dtype=torch.int64)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="a GATHER: the row index comes from a tensor, so the address is "
                 "data-dependent and no affine map describes it. The schedule says so "
                 "by giving that axis no terms rather than inventing one. On this "
                 "hardware a vector gather must read from VTCM (HVX PRM 3.3) -- an ISA "
                 "REQUIREMENT, not a size-derived grant, and this size does not reach "
                 "the staging tier, so the two disagree on purpose",
        ),
    )


class Fp32VarMeanRows(torch.nn.Module):
    def forward(self, x):
        return torch.var_mean(x, dim=-1)


class Fp32StdRows(torch.nn.Module):
    def forward(self, x):
        return torch.std(x, dim=-1)


class Fp32MaxDim(torch.nn.Module):
    def forward(self, x):
        return torch.max(x, dim=-1)


class Fp32Aminmax(torch.nn.Module):
    def forward(self, x):
        return torch.aminmax(x, dim=-1)


class Fp32ArgmaxRows(torch.nn.Module):
    def forward(self, x):
        return torch.argmax(x, dim=-1)


def batch7():
    """Reductions that carry MORE THAN ONE piece of state.

    Everything reduced so far collapsed a row to a single number with a single
    accumulator. These do not: `var_mean` returns two tensors, `max.dim` returns
    the value AND the position that produced it, `aminmax` walks once and keeps
    both ends. That needed structural work rather than five more emitter rows --
    a multi-result node has `shape`/`dtype` of None (there is no single answer),
    which reached `DTYPE_C[...]` as `KeyError: None` and blocked the whole family.
    `Node.results` now carries the per-result pair, `cname` maps the `#` of a
    `producer#k` reference to `_`, and `emit` declares one buffer per result.

    Two things this family makes visible that a single-accumulator reduction
    cannot:

      * **A reduction can be data-dependent.** `argmax`'s inner step is a compare
        plus a conditional update of two registers, and TIES GO TO THE LOWEST
        INDEX -- a strict `>`. Vectorising it needs a running index vector and a
        `vmux` on the compare, not a horizontal `vmax`. `aminmax` is the opposite
        case: two independent accumulators, no data dependence at all, and one
        pass buys both.
      * **A reduction can need TWO passes.** Variance is the mean of squared
        deviations from the mean, so the mean must exist before the second pass
        starts. The single-pass form (E[x^2] - E[x]^2) is algebraically equal and
        numerically much worse, and is not what the reference computes.

    SIZES -- the two-pass pair is the point of this batch. `fp32_var_mean_rows`
    and `fp32_std_rows` compute the SAME statistic with the SAME emitter, and are
    sized on either side of L2 on purpose: at 1.5 MB the second pass re-reads from
    DDR and staging is granted, at 768 KB it re-reads from L2 and staging is not.
    That is the size-derives-the-mechanism rule stated as a controlled pair rather
    than as a claim. Two passes over the operand is also ~4 memory-ops per element,
    which clears the ~3-op staging break-even measured in `ABLATION.md` -- so this
    is the first batch where a DMA/VTCM grant has an arithmetic justification and
    not only a capacity one. The three single-pass kernels are 1-2 ops per element,
    below that break-even, and are sized inside L2 accordingly.
    """
    return (
        KernelSpec(
            name="fp32_var_mean_rows", module=Fp32VarMeanRows(),
            args=(torch.randn(768, 512),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="TWO results and TWO passes. 1.5 MB exceeds L2, so the second pass "
                 "would re-read from DDR -- and a staged tile pays for itself here "
                 "in a way it does not for a one-pass kernel, because the tile is "
                 "read twice for one transfer. Bessel's correction: the divisor is "
                 "n-1, not n",
        ),
        KernelSpec(
            name="fp32_std_rows", module=Fp32StdRows(),
            args=(torch.randn(384, 512),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the SAME two-pass variance, then a square root, at half the size. "
                 "768 KB fits in L2, so the second pass hits cache and no staging is "
                 "granted -- the controlled half of the pair above. `std` decomposes "
                 "to var.correction + sqrt, so it is also the only kernel here whose "
                 "reduction feeds a transcendental",
        ),
        KernelSpec(
            name="fp32_max_dim", module=Fp32MaxDim(),
            args=(torch.randn(512, 448),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="value AND position. The vector form needs a running index vector "
                 "alongside the running max and a vmux on the compare -- and the "
                 "horizontal reduce at the end has to pick the index belonging to the "
                 "winning lane, which a plain vmax discards. Ties go to the lowest "
                 "index",
        ),
        KernelSpec(
            name="fp32_aminmax", module=Fp32Aminmax(),
            args=(torch.randn(512, 448),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="both ends of the range from ONE pass. Two accumulators with no "
                 "dependence between them, which is the contrast to max_dim above: "
                 "same operand, same nest, but nothing data-dependent, so it is two "
                 "ordinary vmin/vmax chains and the pair separates the cost of the "
                 "second accumulator from the cost of the index bookkeeping",
        ),
        KernelSpec(
            name="fp32_argmax_rows", module=Fp32ArgmaxRows(),
            args=(torch.randn(512, 448),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="max_dim with the VALUE discarded -- the index is the whole output. "
                 "Shares max_dim's emitter rather than getting a near-copy, so the "
                 "tie-breaking rule cannot be right in one and wrong in the other. "
                 "int64 indices out of an fp32 reduction, so the result buffer is a "
                 "different width from everything it reads",
        ),
    )


class Fp32CumsumRows(torch.nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=-1)


class Fp32CumsumCols(torch.nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=0)


class Fp32CumprodRows(torch.nn.Module):
    def forward(self, x):
        return torch.cumprod(x, dim=-1)


class Fp32CummaxRows(torch.nn.Module):
    def forward(self, x):
        return torch.cummax(x, dim=-1)


class I32CumsumRows(torch.nn.Module):
    def forward(self, x):
        return torch.cumsum(x, dim=-1)


def batch8():
    """Loops with a CARRIED DEPENDENCE -- the scan family.

    Every nest in batches 1-7 was `parallel` in an axis (iterations independent)
    or `reduction` over it (they all fold to one value). A scan is neither:
    iteration k reads the state iteration k-1 wrote, so reordering changes the
    answer, AND the output keeps the axis at full extent, so nothing is folded
    away. The schedule annotation therefore gains a third iterator kind, `scan`.

    THAT IS NOT THIS MODULE INVENTING A CONCEPT. Linalg has exactly two iterator
    kinds and cannot express a scan at all -- torch-mlir routes cumsum to
    `tm_tensor.scan`, a different dialect, and for `cummax` it fails outright
    ("failed to legalize operation 'torch.operator' ... torch.aten.cummax").
    So batch 8 is the point where the two-kind vocabulary provably runs out, and
    the failing kernel is kept rather than swapped out, because a cross-check
    that reports UNAVAILABLE is evidence and a cross-check quietly avoided is
    not. Labelling a scan `parallel` would licence a vector store across a
    dependence; labelling it `reduction` would licence dropping the axis. Both
    are wrong in the direction that still compiles and still produces plausible
    numbers.

    A scan axis is not vectorisable as a plain loop, but an ASSOCIATIVE scan is
    still a vector algorithm -- shift by 1,2,4,8,16 lanes and combine at each
    step (Hillis-Steele), log2(32) operations for a whole vector's prefix. So
    `mechanism.plan_for` grants HVX off `scan_loops` even though
    `vectorizable_loop()` is None; withholding it would report the entire scan
    family as un-accelerable, which is false.

    THE PAIR THAT MAKES THE POINT: `fp32_cumsum_rows` and `fp32_cumsum_cols` are
    the same op at the same size over the same number of elements, differing
    only in WHICH AXIS carries the dependence. Along the contiguous axis the
    dependence runs through the vector lanes, so the kernel needs five
    shift-and-combine steps per vector plus a serial carry between vectors.
    Along the strided axis every column is independent, so it is an ordinary
    vector accumulate with no shifts at all. Same schedule vocabulary, same
    working set, and the cost difference is entirely the dependence's direction.

    SIZES: 384x320 fp32 throughout -- input plus a same-shape output is 960 KB,
    inside L2, and every kernel here is ~1 operation per element, well below the
    ~3-op staging break-even from `ABLATION.md`. Granting DMA/VTCM to a scan by
    growing it would be granting a mechanism the arithmetic cannot pay for.
    """
    return (
        KernelSpec(
            name="fp32_cumsum_rows", module=Fp32CumsumRows(),
            args=(torch.randn(384, 320),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the dependence runs ALONG the vector, which is the hard "
                 "direction. Five vlalign/add steps give a 32-lane prefix, then a "
                 "scalar carry threads the ten vectors of the row together -- so "
                 "the row is 5x more vector work than a sum and still not a plain "
                 "loop. `vlalign(v, zero, 4n)` is the shift that fills lane i with "
                 "v[i-n] and zero below n (measured on the simulator, not assumed: "
                 "valign shifts the other way)",
        ),
        KernelSpec(
            name="fp32_cumsum_cols", module=Fp32CumsumCols(),
            args=(torch.randn(384, 320),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the SAME op and size with the dependence ACROSS the vector "
                 "instead of along it. Every column is independent, so ten "
                 "accumulator vectors walk the rows and there is no shift, no "
                 "carry, and no horizontal step -- the schedule's scan loop is the "
                 "outer one and the inner one is ordinary. The controlled half of "
                 "the pair: what costs is the dependence's DIRECTION relative to "
                 "the layout, not the dependence",
        ),
        KernelSpec(
            name="fp32_cumprod_rows", module=Fp32CumprodRows(),
            args=(torch.rand(384, 320) * 0.4 + 0.8,),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the same scan with a multiplicative combine, so the shift must "
                 "fill with the MULTIPLICATIVE identity -- vlalign against a splat "
                 "of 1.0f, not against vzero. Inputs are drawn from [0.8, 1.2] "
                 "rather than N(0,1) on purpose: a cumulative product of 320 "
                 "standard normals underflows to exactly 0 within ~40 terms, which "
                 "would make 88% of the comparison vacuous",
        ),
        KernelSpec(
            name="fp32_cummax_rows", module=Fp32CummaxRows(),
            # 256 rows, not 384: this one writes TWO same-shape results, so the
            # working set is 3x the operand and 384 would push it past L2 into a
            # staging grant that ~1 op per element cannot pay for.
            args=(torch.randn(256, 320),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="a scan whose state is a PAIR -- running max and the position it "
                 "came from -- so each Hillis-Steele step is a compare plus two "
                 "vmux. Ties go to the EARLIER index (torch returns the first "
                 "maximal value), which inverts the comparison relative to "
                 "batch 7's max.dim: there the incumbent was earlier, here the "
                 "shifted-in operand is. This is also the kernel torch-mlir cannot "
                 "lower at all",
        ),
        KernelSpec(
            name="i32_cumsum_rows", module=I32CumsumRows(),
            args=(torch.randint(-64, 64, (384, 320), dtype=torch.int32),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the same scan in EXACT arithmetic, and the reason to ship it is "
                 "the correctness argument rather than the code. Hillis-Steele "
                 "reassociates the prefix -- lane 7 is computed as "
                 "((0..3)+(4..5))+(6..7), never left to right -- and for fp32 that "
                 "is a different rounding order covered only by a tolerance. For "
                 "int32 addition is exactly associative, so the vector result is "
                 "BIT-IDENTICAL to the reference and the tolerance argument is not "
                 "needed at all",
        ),
    )


class Fp16Bmm(torch.nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)


class Fp16Addmm(torch.nn.Module):
    def forward(self, bias, a, b):
        return torch.addmm(bias, a, b)


class Fp16MmRelu(torch.nn.Module):
    def forward(self, a, b):
        return torch.relu(a @ b)


class Fp16NormRows(torch.nn.Module):
    def forward(self, x):
        return torch.linalg.vector_norm(x, dim=-1)


class I8MulShift(torch.nn.Module):
    def forward(self, a, b):
        return (a.to(torch.int32) * b.to(torch.int32)) >> 3


def batch9():
    """The CONTRACTION family beyond `mm`, on the narrow datapaths where HMX lives.

    Batches 4-8 were all fp32 and all rank 2. HMX has appeared exactly twice in
    this corpus, both in batch 3, so the entitlement that most needs testing is
    the one with the least evidence behind it. Two new schedule rules make that
    possible -- `aten.bmm` and `aten.addmm`, which `mechanism._contracts` now
    recognises through `MATMUL_TARGETS`:

      * `bmm` is a FOUR-loop nest, [parallel, parallel, parallel, reduction],
        and the batch axis indexes BOTH operands. That is what makes it a
        different nest from an `mm` in a loop, which would re-read one operand
        every iteration -- and the schedule states the difference by giving that
        operand a batch term.
      * `addmm` is the matmul nest with a THIRD operand at stride 0 down the
        rows. Fusing the bias into the contraction instead of emitting a
        separate elementwise pass is what makes the broadcast visible: one splat
        outside the k loop against a whole extra traversal of the (M,N) result.

    THE NEGATIVE CONTROL IS `fp16_norm_rows`, and it earned the role by accident
    -- which is the useful kind. It has a reduction loop, its combine is a SUM OF
    PRODUCTS, and its dtype is fp16: three of the four things the HMX rule looks
    for. What it does not have is the tile-matmul NEST -- one parallel loop, not
    two over a shared reduction -- so `hmx` must be refused, and the batch fails
    as a whole if it is granted. Without a case that must be refused, "the HMX
    rule fires on matmuls" is indistinguishable from "the HMX rule fires".

    (The batch originally used `torch.outer` for this. The provenance stage
    DESTROYED it: outer decomposes to `view` + `mul`, and `view` is harvested as
    layout-only, so it fails `mechanism_eligible` and the whole kernel goes --
    the same tension batch 6 recorded, hit a second time. Recorded here rather
    than worked around, because the workaround would have been to weaken the
    rule while nobody was watching.)

    `fp16_norm_rows` and `i8_mul_shift` are also the two places where the dtype
    of the DATA and the dtype of the ARITHMETIC differ, which is the thing about
    narrow datapaths a fp32 kernel never has to say: a fp16 sum of 512 squares
    saturates the format's ~2048 exact-integer range long before the row ends,
    and an int8 product needs 15 bits so the widening is not optional.

    SIZES: every kernel here is inside L2 (T1). The contractions are the one
    family in this corpus whose arithmetic intensity (K operations per element)
    clears the ~3-op staging break-even by a wide margin, so growing them past
    L2 would be justified -- batch 3 already carries that case, and repeating it
    here would cost tens of minutes of scalar-reference simulation to re-prove a
    point the corpus has. What batch 9 is testing is the GRANT, not the tier.
    """
    return (
        KernelSpec(
            name="fp16_bmm", module=Fp16Bmm(),
            args=(torch.randn(8, 128, 128, dtype=torch.float16),
                  torch.randn(8, 128, 128, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="a four-loop contraction: 8 independent 128x128x128 tile "
                 "matmuls. The batch axis is parallel over BOTH operands, so the "
                 "HMX weight tile has to be reloaded per batch -- unlike a "
                 "broadcast batch, where one load would serve every slice. "
                 "Verifies that the contraction grant survives a nest with more "
                 "parallel loops than `mm` has",
        ),
        KernelSpec(
            name="fp16_addmm", module=Fp16Addmm(),
            args=(torch.randn(256, dtype=torch.float16),
                  torch.randn(256, 256, dtype=torch.float16),
                  torch.randn(256, 256, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="the linear layer: a contraction with the bias FUSED IN as the "
                 "accumulator's initial value rather than added afterwards. The "
                 "bias is rank 1 and stride 0 down the rows, which the schedule "
                 "says by giving its map a column term and no row term -- a "
                 "splat, hoisted out of the k loop",
        ),
        KernelSpec(
            name="fp16_mm_relu", module=Fp16MmRelu(),
            args=(torch.randn(384, 384, dtype=torch.float16),
                  torch.randn(384, 384, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="the first MULTI-PRIMITIVE graph to be granted HMX. Every "
                 "earlier HMX kernel was a single contraction, so the rule had "
                 "only ever been asked `is this op a matmul`; here it has to say "
                 "yes because ONE of two nests contracts. The epilogue is the "
                 "point for the candidate: relu on the HMX readout tile, before "
                 "it is written back, rather than a second full pass over the "
                 "(M,N) result",
        ),
        KernelSpec(
            name="fp16_norm_rows", module=Fp16NormRows(),
            args=(torch.randn(512, 512, dtype=torch.float16),),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="a reduction whose ACCUMULATOR MUST BE WIDER THAN ITS OPERANDS. "
                 "fp16 holds integers exactly only to 2048, and a sum of 512 "
                 "squares of standard normals reaches ~512 -- close enough that "
                 "accumulating in fp16 loses low-order terms outright. The "
                 "candidate has to widen to fp32 (qf32) for the accumulation and "
                 "narrow once, on the store",
        ),
        KernelSpec(
            name="i8_mul_shift", module=I8MulShift(),
            # 256x256, not 512x512: the int32 intermediates make the working set
            # three tensors at FOUR bytes, so 512 lands at 3 MB and would draw a
            # staging grant that a 3-operation elementwise chain cannot pay for.
            args=(torch.randint(-100, 100, (256, 256), dtype=torch.int8),
                  torch.randint(-100, 100, (256, 256), dtype=torch.int8)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the int8 requantise shape: widen to int32, multiply, shift "
                 "right, narrow back. The product of two int8 operands needs 15 "
                 "bits, so the widening is not optional -- and on HVX the widening "
                 "multiply produces a vector PAIR, so one input vector of 128 "
                 "bytes becomes two of 64 lanes each and the narrowing at the end "
                 "has to pack them back. dtype_bytes is 4 because the int32 "
                 "intermediates, not the int8 operands, set the working set",
        ),
    )


class Fp32Rnorm(torch.nn.Module):
    """x / sqrt(x^2 + 1) -- four elementwise primitives, one traversal.

    Chosen for the ladder because its arithmetic intensity is ABOVE the staging
    break-even: square, add, rsqrt (a square root and a divide) and multiply is
    roughly five to eight vector operations per element against the ~3 measured
    in ABLATION.md. A ladder built on a 1-op kernel would grant DMA at the top
    and the honest answer would be that the grant does not pay -- which would
    make the ladder a demonstration of the rule being wrong.
    """

    def forward(self, x):
        return x * torch.rsqrt(x * x + 1.0)


def batch10():
    """THE TIER LADDER: one kernel, five sizes, nothing else varied.

    Every batch so far has argued that the mechanism is DERIVED FROM THE SIZE.
    Each of those batches varied the operation at the same time, so the argument
    always rested on comparing kernels that differed in two ways at once. Batch 7
    got closer with var_mean/std (same statistic, two sizes, two grants) and
    batch 8 closer still with cumsum rows/cols (same op, same size, one axis
    swapped). This batch removes the last confound: the SAME MODULE at five
    sizes, so the only thing that changes down the table is the number of
    elements, and the mechanism column changes with it.

        rung   shape        working set   tier   granted
        T0     24 x 64          12,288 B   T0    hvx
        T1a    64 x 256        131,072 B   T1    hvx + l2fetch
        T1b    256 x 512     1,048,576 B   T1    hvx + l2fetch
        T2     768 x 1024    6,291,456 B   T2    hvx + l2fetch + dma + vtcm
        T3     1280 x 1024  10,485,760 B   T3    hvx + l2fetch + dma + vtcm

    THREE THINGS THE LADDER IS FOR, and only the first is the obvious one:

      * T0 IS A NEGATIVE CONTROL FOR THE MEMORY SIDE, the counterpart to batch
        9's `fp16_outer` for HMX. 12 KB fits in L1D, so l2fetch is NOT granted
        and a candidate that issues one is using a mechanism the size does not
        justify. Until this batch the corpus had no kernel small enough to
        refuse anything, so `l2fetch` had never been observed being withheld.
      * T1a AND T1b ARE THE SAME GRANT EIGHT SIZES APART. A rule that produced a
        new mechanism at every rung would be a rule fitted to the ladder. These
        two exist so the table has a step where the answer does not change.
      * T2 AND T3 ARE THE SAME GRANT AND A DIFFERENT KERNEL. Both exceed L2 so
        both get DMA and VTCM; only T3 exceeds VTCM, so only T3 CANNOT hold its
        working set in the scratchpad and must stream in tiles with double
        buffering. `plan_for` records that distinction as a second `dma` reason
        rather than as a mechanism, which is right -- it is the same hardware
        used a different way -- but it means the mechanism column alone does not
        tell a candidate author which of the two kernels they are writing. That
        is a real limitation of the flag vocabulary and the ladder is where it
        becomes visible.
    """
    return (
        KernelSpec(
            name="fp32_rnorm_t0", module=Fp32Rnorm(),
            args=(torch.randn(24, 64),),
            dtype_bytes=4, expect_tier="T0",
            expect_mechanisms=frozenset({"hvx"}),
            note="12 KB, inside the 16 KB L1D. The only kernel in this corpus "
                 "that is granted NO memory mechanism at all: there is nothing "
                 "for a prefetch to hide because the whole working set is already "
                 "as close as it can get. A candidate that issues an l2fetch here "
                 "is not faster, it is exceeding its entitlement",
        ),
        KernelSpec(
            name="fp32_rnorm_t1a", module=Fp32Rnorm(),
            args=(torch.randn(64, 256),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="128 KB: the first rung past L1D, so the stream misses L1 and "
                 "prefetch has something to hide. Nothing else about the kernel "
                 "changed from T0 -- same module, same four primitives, same "
                 "vector loop -- which is the point",
        ),
        KernelSpec(
            name="fp32_rnorm_t1b", module=Fp32Rnorm(),
            args=(torch.randn(256, 512),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="1 MB, exactly L2, and deliberately the SAME grant as T1a at "
                 "eight times the size. The tier boundary is inclusive at the top "
                 "-- a working set of exactly L2 still fits in L2 -- so this is "
                 "the last size before staging is justified, and the rung that "
                 "shows the rule is not simply issuing a new mechanism per row",
        ),
        KernelSpec(
            name="fp32_rnorm_t2", module=Fp32Rnorm(),
            args=(torch.randn(768, 1024),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="6 MB: past L2, inside VTCM. The data no longer stays resident, "
                 "so staging is what buys bandwidth -- and at five-to-eight "
                 "operations per element this kernel clears the ~3-op break-even "
                 "from ABLATION.md, so the grant is one the arithmetic can pay "
                 "for. The whole working set fits the scratchpad, so a single "
                 "residency is enough",
        ),
        KernelSpec(
            name="fp32_rnorm_t3", module=Fp32Rnorm(),
            args=(torch.randn(1280, 1024),),
            dtype_bytes=4, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="10 MB: past VTCM too. Same mechanism set as T2 and a DIFFERENT "
                 "kernel -- the scratchpad can no longer hold the working set, so "
                 "the tiles have to be streamed and double-buffered, with the "
                 "transfer of tile k+1 running under the compute on tile k. The "
                 "flag vocabulary cannot express that difference; `plan_for` "
                 "records it as a second reason on `dma`",
        ),
    )


class Fp32SortRows(torch.nn.Module):
    def forward(self, x):
        return torch.sort(x, dim=-1)


class Fp32TopkRows(torch.nn.Module):
    def forward(self, x):
        return torch.topk(x, 8, dim=-1)


class Fp32MedianRows(torch.nn.Module):
    def forward(self, x):
        return torch.median(x, dim=-1)


class Fp32KthvalueRows(torch.nn.Module):
    def forward(self, x):
        return torch.kthvalue(x, 40, dim=-1)


class Fp32ClampTensor(torch.nn.Module):
    def forward(self, x, b):
        lim = b.abs()
        return torch.clamp(x, -lim, lim)


def batch11():
    """ORDER STATISTICS -- the answer depends on a COMPARISON NETWORK.

    Every kernel so far computed each output from a fixed set of inputs: an
    elementwise op reads one, a reduction folds a row, a scan threads a
    dependence. None of them had to know how their inputs COMPARE to each other.
    Sorting does, and that changes what the schedule can say.

    THE SORTED AXIS IS MARKED `reduction`, AND THE OTHER TWO LABELS ARE BOTH
    WRONG. Not `parallel`: the output at sorted position p depends on the whole
    row, not on input position p, so there is no per-iteration independence to
    vectorise across. Not `scan` either, which is the interesting one -- a scan's
    iteration k reads what k-1 wrote, a strictly sequential chain, whereas a
    comparison network has no chain at all: bitonic sort is log^2(n) DATA-PARALLEL
    stages. `reduction` is the closest of the three, and it makes `plan_for` grant
    HVX through the accumulator clause rather than the vector-store clause, which
    is the honest entitlement: these vectorise, but not as a unit-stride loop over
    the sorted axis.

    FOUR OPS, ONE SORT. `sort`, `topk`, `median.dim` and `kthvalue` differ only in
    which elements of the sorted row they keep, so they share one emitter --
    otherwise the stability rule and the index bookkeeping get four chances to
    disagree with each other. The sort is insertion sort with a strict `>`, which
    never moves an element past an equal one, so equal values keep their original
    order and the INDEX output is deterministic. torch does not promise a stable
    sort by default; an unstable reference would be irreproducible against its own
    golden.

    THE REFERENCE IS O(n^2) AND THE CANDIDATES ARE NOT, so read the ratios with
    that in mind. A naive scalar sort is quadratic -- that is what "naive scalar
    reference" means here -- while the candidates are log^2(n) networks and O(n)
    counting passes. Part of every ratio in this batch is an ALGORITHMIC change
    rather than a hardware one, exactly as batch 9's fp16 matmuls were inflated by
    the reference's software-emulated fp16. Stated here rather than left for a
    reader to discover.

    `aten.median.dim` is the SECOND op torch-mlir cannot lower at all (after
    `aten.cummax` in batch 8) -- it is explicitly marked illegal. Kept rather than
    swapped out, with a `no-linalg` verdict carrying the compiler's own
    diagnostic: a cross-check that reports UNAVAILABLE is evidence, and one
    quietly avoided is not.

    SIZES: rows are 128 elements. Two reasons, and neither is cosmetic -- the
    emitted scratch row lives on the stack, and the reference is quadratic in the
    row, so a 1024-wide row would cost 64x the reference time for no extra
    information. All five are inside L2 at ~1 comparison per element pair, so no
    staging is granted.
    """
    return (
        KernelSpec(
            name="fp32_sort_rows", module=Fp32SortRows(),
            args=(torch.randn(256, 128),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the full permutation, values AND indices. A bitonic network is "
                 "the vector form: log2(128)*(log2(128)+1)/2 = 28 compare-exchange "
                 "stages, each a full-width vector operation, against the "
                 "reference's ~8,128 scalar comparisons per row. The indices have "
                 "to ride along through every exchange, which is what makes it "
                 "more than a sort of values",
        ),
        KernelSpec(
            name="fp32_topk_rows", module=Fp32TopkRows(),
            args=(torch.randn(256, 128),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the 8 largest, in descending order. A full sort is not needed "
                 "and the vector form is 8 masked argmax passes -- find the max, "
                 "mask the winner out, repeat -- so the cost is k*n rather than "
                 "n log^2 n. The pair with fp32_sort_rows isolates what SORTING "
                 "costs beyond SELECTING",
        ),
        KernelSpec(
            name="fp32_median_rows", module=Fp32MedianRows(),
            args=(torch.randn(256, 128),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the LOWER median (sorted position (n-1)/2) and its index. No "
                 "sort required at all: binary-search the 32-bit monotone key "
                 "space, counting elements below each candidate with a vector "
                 "compare -- 32 iterations of n/32 vector ops, so O(n) per row "
                 "against the reference's O(n^2). This is also the op torch-mlir "
                 "refuses to lower",
        ),
        KernelSpec(
            name="fp32_kthvalue_rows", module=Fp32KthvalueRows(),
            args=(torch.randn(256, 128),),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the 40th smallest and its index -- the SAME machinery as the "
                 "median with a different target rank, which is the point of "
                 "shipping both. If the median kernel were secretly exploiting "
                 "the midpoint (a two-sided partition, say) this one would not "
                 "work, and the pair is what shows it does not",
        ),
        KernelSpec(
            name="fp32_clamp_tensor", module=Fp32ClampTensor(),
            # 256x256, not 256x384: the chain materialises abs, neg and two
            # multiplies alongside the two inputs and the result, so 384 columns
            # push the working set past L2 into a staging grant that a 4-operation
            # elementwise kernel cannot pay for.
            args=(torch.randn(256, 256), torch.randn(256, 256) * 0.5),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the elementwise control: three operands, two of them derived "
                 "bounds, and no comparison BETWEEN elements -- only between an "
                 "element and its own bound. `clamp.Tensor` needs a different "
                 "emitter from `clamp.default` because there an absent bound is a "
                 "missing ARGUMENT, while here both bounds are real tensors and "
                 "the ordinary three-operand elementwise form applies",
        ),
    )


class Fp16MmSoftmax(torch.nn.Module):
    def forward(self, a, b):
        return torch.softmax(a @ b, dim=-1)


class Fp32Softmax4d(torch.nn.Module):
    def forward(self, x):
        return torch.softmax(x, dim=-1)


class Fp32ScaleChannel4d(torch.nn.Module):
    def forward(self, x, s):
        return x * s


class Fp32AmaxHW(torch.nn.Module):
    def forward(self, x):
        return torch.amax(x, dim=(2, 3))


class Fp32VarHW(torch.nn.Module):
    def forward(self, x):
        return torch.var(x, dim=(2, 3))


def batch12():
    """RANK 4, and the two gaps the corpus-wide EDA found.

    An audit of all 55 kernels turned up two specific holes, and this batch is
    built to fill them rather than to explore a new op family:

      * **T3 was thin** -- 5 kernels out of 55 exceeded VTCM, against 31 sitting
        in T1. Three kernels here are at or past the VTCM boundary.
      * **`l2fetch` + `dma`/`vtcm` together had never been USED by any candidate.**
        11 kernels are entitled to both and every one of them drops the prefetch,
        correctly: the V75 HVX PRM wants an l2fetch target L2-cacheable (3.9.4)
        and VTCM data marked uncached (3.4), so for the SAME data the two are
        alternatives. `fp16_mm_softmax` is the case where they are not competing,
        because they apply to DIFFERENT operands -- which is the combined case
        `RESUME.md` has had open as unmeasured since batch 3.

    RANK 4 IS ALSO WHERE THE INDEXING MAPS EARN THEIR KEEP. Until now the only
    kernel with more than three loops was convolution. At rank 4 the schedule has
    to say things a rank-2 nest never needs to:

      * `fp32_scale_channel_4d` multiplies an (8,16,64,128) tensor by a
        (1,16,1,1) one, so THREE of the operand's four axes are pinned to
        constants. That is a stride-0 broadcast in three dimensions at once --
        one vector load feeds 8*64*128 = 65,536 output elements per channel.
      * `fp32_amax_hw` and `fp32_var_hw` reduce over TWO axes and keep two, so
        the nest is ['parallel','parallel','reduction','reduction'] and the
        result map drops two of four.

    THE BOUNDARY PAIR. `fp32_softmax_4d` and `fp32_scale_channel_4d` hold the
    same 4 MB tensor and produce the same 4 MB result, so both working sets are
    8,388,608 bytes -- exactly VTCM. The scale kernel also reads a 64-byte
    channel vector, which puts it at 8,388,672, and THAT tips it from T2 to T3.
    The tier boundary is inclusive at the top (a working set of exactly VTCM
    still fits in VTCM), so 64 bytes is the difference between "stage it once"
    and "stream it in tiles". A rule stated as a threshold has to be tested at
    the threshold; this is the pair that does it.
    """
    return (
        KernelSpec(
            name="fp16_mm_softmax", module=Fp16MmSoftmax(),
            args=(torch.randn(1024, 32, dtype=torch.float16),
                  torch.randn(32, 512, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T2",
            expect_mechanisms=frozenset({"hmx", "hvx", "dma", "vtcm", "l2fetch"}),
            note="THE COMBINED CASE, and the only kernel in this corpus entitled "
                 "to all five mechanisms. The weight operand is 32 KB and every "
                 "row strip reads all of it, so it wants to be L2-resident and "
                 "prefetched; the 1 MB result is written once and never re-read, "
                 "so it wants a staged VTCM buffer drained by DMA. Those are "
                 "DIFFERENT operands, which is why the PRM's "
                 "l2fetch-or-VTCM-but-not-both rule does not bite here. K=32 is "
                 "exactly one HMX tile deep, so the contraction is a single "
                 "activation/weight pair per output tile",
        ),
        KernelSpec(
            name="fp32_softmax_4d", module=Fp32Softmax4d(),
            args=(torch.randn(8, 16, 64, 128),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="a rank-4 nest whose reduction is the innermost axis: three "
                 "parallel loops over 8*16*64 = 8,192 independent rows of 128. "
                 "The working set is 8,388,608 bytes -- EXACTLY VTCM -- so it is "
                 "T2 and one residency suffices. Its pair below is the same "
                 "tensor plus 64 bytes and is T3",
        ),
        KernelSpec(
            name="fp32_scale_channel_4d", module=Fp32ScaleChannel4d(),
            args=(torch.randn(8, 16, 64, 128), torch.randn(1, 16, 1, 1)),
            dtype_bytes=4, expect_tier="T3",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="a per-channel scale: THREE of the four axes of the second "
                 "operand are pinned to constants, so one loaded value feeds "
                 "65,536 output elements. The schedule states it as an operand "
                 "whose map has a term for d1 and nothing for d0, d2 or d3. It is "
                 "also the T3 half of the boundary pair -- 64 extra bytes of "
                 "channel vector on an 8,388,608-byte working set is what pushes "
                 "it past VTCM and turns one residency into a tiled stream",
        ),
        KernelSpec(
            name="fp32_amax_hw", module=Fp32AmaxHW(),
            args=(torch.randn(8, 32, 64, 96),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="reduce over TWO axes and keep two: the spatial max per (batch, "
                 "channel), which is the global-pooling shape. The nest is "
                 "['parallel','parallel','reduction','reduction'] and the result "
                 "map drops two of four axes -- the first kernel here whose "
                 "reduction is more than one loop deep and is not a matmul",
        ),
        KernelSpec(
            name="fp32_var_hw", module=Fp32VarHW(),
            args=(torch.randn(8, 16, 64, 64),),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="the same two-axis reduction, TWICE over: spatial variance needs "
                 "the mean before the deviations, so each (batch, channel) plane "
                 "is traversed twice. That is batch 7's var_mean argument at rank "
                 "4 -- and past L2, so the second pass is exactly what the staged "
                 "VTCM plane buys. torch-mlir emits one reduction nest per pass, "
                 "which is why this reports `multi-pass` rather than EXACT",
        ),
    )


class U8RequantPerChannel(torch.nn.Module):
    """int32 accumulator -> uint8, per-channel multiplier and a rounding shift.

    Written in plain tensor ops because `torch.quantize_per_channel` does not
    survive `torch.export`, and this is the arithmetic it performs.
    """

    def forward(self, acc, mult):
        # int64 for the product, NOT int32. torch's int32*int32 WRAPS, and a
        # requant multiplier is Q15, so acc*mult needs up to 47 bits: at
        # acc=200000, mult=40000 the true product is 8e9 and int32 gives
        # -589,934,592. A reference built on that would have a golden made of
        # wrapped garbage, and a kernel matching it would be matching the bug.
        p = acc.to(torch.int64) * mult.to(torch.int64)
        return torch.clamp((p + (1 << 14)) >> 15, 0, 255).to(torch.uint8)


class I8RequantSymmetric(torch.nn.Module):
    def forward(self, acc, mult):
        p = acc.to(torch.int64) * mult.to(torch.int64)   # see the uint8 sibling
        return torch.clamp((p + (1 << 14)) >> 15, -128, 127).to(torch.int8)


class U8DequantPerChannel(torch.nn.Module):
    def forward(self, q, scale, zero):
        return (q.to(torch.int32) - zero) * scale


class I8AddRequant(torch.nn.Module):
    """The residual add: two int8 tensors on DIFFERENT scales, rescaled to a
    common one and requantised. Both multipliers are compile-time constants here
    because a real graph folds the two input scales and the output scale into
    exactly two integers."""

    def forward(self, a, b):
        # int32 is safe HERE: int8 operands times single-digit multipliers reach
        # at most 127*5*2 = 1,270. The int64 promotion the requant kernels need is
        # about the SIZE of a Q15 multiplier, not about int8 arithmetic.
        s = a.to(torch.int32) * 3 + b.to(torch.int32) * 5
        return torch.clamp((s + (1 << 3)) >> 4, -128, 127).to(torch.int8)


class F32ToU8Quantize(torch.nn.Module):
    def forward(self, x, scale):
        return torch.clamp(torch.round(x / scale), 0, 255).to(torch.uint8)


def batch13():
    """THE QUANTISATION PIPELINE -- int8 arithmetic on HVX, because HMX cannot.

    The corpus-wide EDA found the int8 slice thin: 5 kernels of 55, all simple,
    and ZERO of them reaching HMX. `run_artifacts/forge2/HMX_INT8.md` records why
    that is not a gap anyone can close from here -- four probe rounds established
    that the int8 tile path's readout works (`out = bias_high >> 7` exactly) while
    the accumulate contributes nothing, across five addressing variants, four tile
    masks, and with or without the conversion and shift steps. HMX int8 is
    undocumented in all 29 shipped PDFs and could not be derived.

    So this batch takes the honest position: an NPU-shaped int8 pipeline built on
    the vector unit. These are the five ops a real quantised graph is actually made
    of, between its convolutions, and they are written in plain tensor ops because
    `torch.quantize_per_channel` does not survive `torch.export`.

    WHAT MAKES THEM DIFFERENT FROM THE fp32 ELEMENTWISE KERNELS, and it is not the
    dtype on its own:

      * **The arithmetic is WIDER than both the input and the output.** An int32
        accumulator times a per-channel multiplier needs 64 bits before the shift
        brings it back to 8. On HVX that means the widening multiply produces a
        vector PAIR and the narrowing at the end has to pack two of them back --
        the same deal-order bookkeeping batch 9's `i8_mul_shift` had, now with a
        rounding term and a saturating clamp on the end.
      * **Rounding is part of the specification, not a detail.** `+ (1 << 14)`
        before `>> 15` is round-half-up, and `torch.round` is round-half-to-EVEN.
        Those differ on exactly the ties, and a kernel that picks the other one is
        wrong on a measurable fraction of any real tensor.
      * **Saturation is not clamping to the dtype's range afterwards.** The clamp
        happens in int32 and the narrowing cast follows; doing it the other way
        round wraps instead of saturating, which is the classic quantisation bug
        and looks like noise rather than like an error.

    SIZES span T0 to T2, and `i8_requant_symmetric` at 24x64 is deliberately T0 --
    12,544 bytes, inside the 16 KB L1D, so it is granted NO memory mechanism at
    all. The EDA found only 3 such kernels in 55, which meant `l2fetch` had almost
    never been observed being WITHHELD. This is a fourth.
    """
    return (
        KernelSpec(
            name="u8_requant_per_channel", module=U8RequantPerChannel(),
            args=(torch.randint(-200000, 200000, (256, 256), dtype=torch.int32),
                  torch.randint(16384, 32768, (1, 256), dtype=torch.int32)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the op every quantised convolution ends with. The multiplier is "
                 "per-CHANNEL, so it is a stride-0 broadcast down the rows -- one "
                 "vector load reused for all 256 rows. int32 x int32 needs 64 bits, "
                 "so the multiply widens to a vector pair and the >>15 with a "
                 "round-half-up bias happens before the pair is narrowed and "
                 "saturated to uint8",
        ),
        KernelSpec(
            name="i8_requant_symmetric", module=I8RequantSymmetric(),
            args=(torch.randint(-200000, 200000, (24, 64), dtype=torch.int32),
                  torch.randint(16384, 32768, (1, 64), dtype=torch.int32)),
            dtype_bytes=4, expect_tier="T0",
            expect_mechanisms=frozenset({"hvx"}),
            note="the same arithmetic with a SYMMETRIC output range and, at 12,544 "
                 "bytes, inside L1D -- so it is granted no memory mechanism and a "
                 "candidate that prefetches is exceeding its entitlement. Only the "
                 "clamp bounds differ from the uint8 sibling, which is the point: "
                 "signed and unsigned quantisation are the same kernel with two "
                 "constants changed, and the pair says so",
        ),
        KernelSpec(
            name="u8_dequant_per_channel", module=U8DequantPerChannel(),
            args=(torch.randint(0, 255, (384, 256), dtype=torch.uint8),
                  torch.randn(1, 256),
                  torch.randint(0, 255, (1, 256), dtype=torch.int32)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the inverse direction: uint8 -> fp32 as (q - zero_point) * scale, "
                 "both per-channel. TWO stride-0 operands in one kernel, one "
                 "integer and one float, and the dtype changes mid-graph -- the "
                 "subtract is integer, the multiply is float, so the widening from "
                 "byte to word and the conversion from word to float are both in "
                 "the same traversal",
        ),
        KernelSpec(
            name="i8_add_requant", module=I8AddRequant(),
            args=(torch.randint(-128, 127, (384, 512), dtype=torch.int8),
                  torch.randint(-128, 127, (384, 512), dtype=torch.int8)),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="the RESIDUAL ADD, which is the one place a quantised network has "
                 "to reconcile two different scales: each input is rescaled by its "
                 "own integer multiplier into a common int32 domain, summed, then "
                 "requantised once. Nine primitives and the longest int8 chain in "
                 "the corpus. Past L2, so staging is granted -- and with two int8 "
                 "inputs widening to int32 the intermediate traffic is four times "
                 "the operand traffic, which is what makes the tier reading larger "
                 "than the tensors suggest",
        ),
        KernelSpec(
            name="f32_to_u8_quantize", module=F32ToU8Quantize(),
            args=(torch.randn(320, 256), torch.rand(1, 256) + 0.5),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="the graph's entry point: fp32 -> uint8 with a per-channel scale. "
                 "`torch.round` is round-half-to-EVEN, not half-away, and the "
                 "difference shows up on exactly the ties -- HVX has no float "
                 "round-to-nearest-even instruction, so the candidate needs the "
                 "add-magic-constant trick (add 1.5*2^23, subtract it back), which "
                 "inherits the FPU's current mode and is even by default",
        ),
    )


class Fp16Conv1x1(torch.nn.Module):
    def forward(self, x, w):
        return torch.conv2d(x, w)


class Fp16Conv3x3Stride2(torch.nn.Module):
    def forward(self, x, w):
        return torch.conv2d(x, w, stride=2)


class Fp16Conv3x3Dilation2(torch.nn.Module):
    def forward(self, x, w):
        return torch.conv2d(x, w, dilation=2)


class Fp16Conv3x3Pad1(torch.nn.Module):
    def forward(self, x, w):
        return torch.conv2d(x, w, padding=1)


class Fp16ConvDepthwise(torch.nn.Module):
    def forward(self, x, w):
        return torch.conv2d(x, w, groups=32)


def batch14():
    """THE CONVOLUTION FAMILY -- the only nest that needs the general affine form.

    Every other kernel in this corpus indexes its operands with "loop k walks axis
    a, coefficient 1". Convolution cannot: its input height is
    `oh*stride + kh*dilation`, TWO loops with coefficients indexing ONE axis, which
    is why `schedule.Term` exists at all. That machinery has been exercised by
    exactly two kernels (batch 3's dense and depthwise conv) in one configuration
    each. This batch varies the configuration and nothing else, so what the maps
    say can be checked against what the hardware can do.

    THE RESULT WORTH HAVING: STRIDE AND DILATION ARE NOT THE SAME KIND OF THING,
    though both are colloquially "a strided convolution". Measured
    `vectorizable_loop` across the five:

        conv1x1                 vec = d3 (ow)
        conv3x3 stride 2        vec = NONE
        conv3x3 dilation 2      vec = d3 (ow)
        conv3x3 padding 1       vec = d3 (ow)
        depthwise               vec = d3 (ow)

    Stride multiplies the OUTPUT loop's coefficient on the input axis, so at
    stride 2 consecutive `ow` steps land two elements apart and the contiguous
    vector load is gone -- the loop is a gather and the schedule refuses it.
    Dilation multiplies the KERNEL loop's coefficient instead, and `kw` is not the
    vectorised axis, so `ow` stays unit-stride and the load survives. The
    annotation gets this right for the reason it was designed to: it tracks
    coefficients per (loop, axis) pair rather than recording "this convolution is
    strided". A table that stored one flag per convolution could not distinguish
    these two cases, and both would have to be pessimised.

    HMX IS GRANTED TO FOUR AND REFUSED TO ONE. A convolution contracts over
    (ic, kh, kw), which is the tile-matmul shape, so `_contracts` grants `hmx`
    when groups == 1 and the dtype fits. Depthwise (groups == Cout) reduces over
    kh/kw WITHIN one channel and contracts nothing across channels, so there is no
    tile matmul to map onto and the grant is withheld. That is the negative
    control, and it is a different reason from batch 1's (wrong dtype) and batch
    9's (wrong nest) -- three independent ways for the same rule to say no.

    A DOCUMENTED BOUNDARY, not a kernel: general grouped convolution with
    1 < groups < Cout is UNSCHEDULABLE here and raises rather than guessing. The
    input channel index becomes `(oc / (Cout/groups))*Cin_g + ic`, a floordiv of a
    loop, which is not an affine expression and cannot be written as a sum of
    `Term`s at all. groups == 1 and groups == Cout are the two exact cases. The
    honest options are to extend the representation or to exclude the op; it
    currently excludes, loudly.

    SIZES: all five are T1, and the WIDTHS are chosen so that every output row is
    exactly 64 halfwords -- one HVX vector. That is not cosmetic. This header
    declares `Q6_Q_vsetq_R` for building a lane predicate and NO masked vector
    store to consume it, so a row that is not a whole number of vectors needs
    either a read-modify-write across the row boundary or a scalar tail; at
    OW <= 32 the tail would be the entire row. Padding the shape is what a real
    deployment does too. The input widths that produce OW = 64 are therefore 64,
    130, 68, 64 and 66 respectively -- the stride-2 case needs 130 because
    (130-3)/2 + 1 = 64.

    Convolution is the most arithmetic-dense op in the corpus -- 9 taps times Cin
    per output element, 1 to 19 MMACs here -- and this batch is about the MAPS, so
    the sizes keep the scalar references cheap rather than reaching a staging tier
    batch 3 already covers.
    """
    return (
        KernelSpec(
            name="fp16_conv1x1", module=Fp16Conv1x1(),
            args=(torch.randn(2, 32, 32, 64, dtype=torch.float16),
                  torch.randn(64, 32, 1, 1, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="the degenerate convolution: a 1x1 kernel has no spatial extent, "
                 "so kh and kw are single-iteration loops and the nest collapses to "
                 "a contraction over input channels at every pixel -- a matmul in "
                 "disguise, with (n, oh, ow) as the row index. The maps say so: "
                 "with KH=KW=1 the input's height term loses its kh contribution "
                 "entirely and becomes just oh. Worth having because it is the "
                 "convolution whose schedule a reader would most expect to be "
                 "wrong",
        ),
        KernelSpec(
            name="fp16_conv3x3_stride2", module=Fp16Conv3x3Stride2(),
            args=(torch.randn(2, 16, 32, 130, dtype=torch.float16),
                  torch.randn(32, 16, 3, 3, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="STRIDE 2, and `vectorizable_loop` is NONE. Consecutive ow steps "
                 "advance the input by two elements, so there is no contiguous "
                 "vector load along the output's fastest axis -- the loop is a "
                 "gather. The candidate must either gather or vectorise a different "
                 "axis, and the schedule is what says which. Verified against "
                 "linalg-generalize-named-ops, whose printed map for "
                 "conv_2d_nchw_fchw at stride 2 is (d0, d4, d2*2 + d5, d3*2 + d6)",
        ),
        KernelSpec(
            name="fp16_conv3x3_dilation2", module=Fp16Conv3x3Dilation2(),
            args=(torch.randn(2, 16, 32, 68, dtype=torch.float16),
                  torch.randn(32, 16, 3, 3, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="DILATION 2, and `vectorizable_loop` is d3 -- the contrast that "
                 "makes the stride kernel meaningful. Dilation scales the KERNEL "
                 "loop's coefficient, spreading the taps across the input, and "
                 "leaves the output loop at coefficient 1. So ow is still "
                 "unit-stride and the vector load survives; what changes is that "
                 "the three taps of a row are 2 elements apart instead of 1, which "
                 "is a different vector LOAD PATTERN and not a different loop "
                 "structure",
        ),
        KernelSpec(
            name="fp16_conv3x3_pad1", module=Fp16Conv3x3Pad1(),
            args=(torch.randn(2, 16, 32, 64, dtype=torch.float16),
                  torch.randn(32, 16, 3, 3, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="PADDING 1, so the output keeps the input's 32x32 shape and every "
                 "tap needs a bounds check -- the input index oh + kh - 1 is "
                 "out of range on the first and last row and column. The affine map "
                 "is unchanged by padding (it is still oh + kh with an offset), so "
                 "this is the case where the schedule is NOT the whole story: the "
                 "map describes where a tap reads, and the padding decides whether "
                 "that read happens at all. The candidate's real work is making the "
                 "interior branch-free and handling the border separately",
        ),
        KernelSpec(
            name="fp16_conv_depthwise", module=Fp16ConvDepthwise(),
            args=(torch.randn(2, 32, 32, 66, dtype=torch.float16),
                  torch.randn(32, 1, 3, 3, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="THE NEGATIVE CONTROL: groups == Cout, so each output channel sees "
                 "exactly one input channel. The nest still has parallel loops over "
                 "a reduction, and the dtype is fp16, and `hmx` is still REFUSED -- "
                 "because the reduction is over kh/kw within a single channel and "
                 "nothing is contracted ACROSS channels, so there is no tile matmul "
                 "to map onto. Granting it would create an entitlement no correct "
                 "kernel could satisfy, which then reads as a model failure in the "
                 "results table",
        ),
    )


def _with_nonfinite(t):
    """Plant NaN, +inf and -inf in a tensor, deterministically.

    `nan_to_num` is the only task here whose input needs them, and every other
    task draws from a normal distribution -- so without this the isnan/isinf
    emitters are compiled and never executed on a value that reaches them. The
    positions are fixed rather than random so the golden is reproducible.
    """
    t = t.clone()
    flat = t.view(-1)
    flat[0] = float("nan")
    flat[1] = float("inf")
    flat[2] = float("-inf")
    flat[len(flat) // 2] = float("nan")
    flat[-1] = float("-inf")
    return t


class Fp32Lerp(torch.nn.Module):
    def forward(self, a, b, w):
        return torch.lerp(a, b, w)


class Fp32Addcmul(torch.nn.Module):
    def forward(self, t, a, b):
        return torch.addcmul(t, a, b, value=0.5)


class Fp32Addcdiv(torch.nn.Module):
    def forward(self, t, a, b):
        return torch.addcdiv(t, a, b, value=0.5)


class Fp16Baddbmm(torch.nn.Module):
    def forward(self, t, a, b):
        return torch.baddbmm(t, a, b)


class Fp32SanitizeNonfinite(torch.nn.Module):
    """Replace every non-finite element of `x` with the matching element of `y`.

    `torch.nan_to_num` was the first choice and the provenance stage DESTROYED it:
    it decomposes through `aten.scalar_tensor`, which is harvested as
    creation-not-arithmetic and so fails `mechanism_eligible`. Third batch in a row
    to lose a kernel that way (6, 9, 15). Substituting a tensor rather than a
    literal keeps the whole graph eligible -- isnan, isinf, bitwise_or and
    where.self are all arithmetic -- and keeps the point: the input really does
    contain NaN and both infinities.
    """

    def forward(self, x, y):
        return torch.where(torch.isnan(x) | torch.isinf(x), y, x)


def batch15():
    """THREE-OPERAND OPS -- the bucket the PROBER extension unlocked.

    This batch is chosen by an instrument rather than by taste, which is the thing
    `coverage.py` exists for. Its four-way split said 687 of 1,228 eligible rows
    were `unprobeable` -- the prober could not even call them -- and a breakdown of
    WHY put the largest group at "3 required arguments" (98 rows), followed by a
    Scalar argument (88) and 4 required arguments (53). The prober accepted only
    1-2 plain Tensors, so `where`, `addcmul`, `addcdiv`, `lerp` and `baddbmm` --
    ordinary, useful ops -- had never been counted at all.

    `synth_signature` now builds a per-position argument plan and reaches **341
    more (op, overload) pairs**. The five kernels here are drawn from that set.

    AND THE EXTENSION IS OFF BY DEFAULT, because measuring it produced a SEGFAULT.
    The rows it newly reaches include `_ctc_loss`, `_cholesky_solve_helper`,
    `_convert_weight_to_int4pack` and `_dyn_quant_matmul_4bit` -- kernels with real
    preconditions among their arguments, handed independent guesses. A bad value
    there does not raise, it takes the interpreter down, and a prober that dies
    produces no number at all. Making it safe means one subprocess per probe, which
    is a design change and not a patch. So `ALLOW_SYNTH_DEFAULT = False`: the
    default scan behaves exactly as before, `--allow-synth` opts in, and the
    segfault is documented at the top of the synthesiser rather than left for the
    next person to rediscover.

    WHAT THREE OPERANDS ACTUALLY COSTS, which is the batch's own question. A
    two-operand elementwise kernel reads two streams and writes one. These read
    three and write one, so the load:store ratio goes from 2:1 to 3:1 and the
    kernels are further into bandwidth-bound territory at the same element count.
    Two of them cross into T2 on that alone -- `addcmul` at 256x384 is 1.5 MB of
    working set for what looks like a 384 KB tensor.

    `fp32_lerp` is the interesting one for a different reason: it is EIGHT
    primitives. torch decomposes a tensor-weighted lerp into a comparison, two
    subtractions, a multiply, a division and a `where`, because the numerically
    stable form differs depending on whether the weight is above or below 0.5. A
    candidate that writes the naive `a + w*(b-a)` computes something slightly
    different from the reference and, at a 1e-3 relative tolerance, gets away with
    it -- so the kernel follows the decomposition instead. The graph is the
    specification even when a shorter formula looks equivalent.

    `fp32_sanitize_nonfinite` is the only kernel in this corpus whose INPUT
    contains NaN and infinity on purpose. Everything else is drawn from a normal
    distribution, so the `isnan`/`isinf` emitters added in batch 4 have been
    compiled sixty times and never once given a value that reaches them.

    It was `torch.nan_to_num` until the provenance stage DESTROYED it -- that
    decomposes through `aten.scalar_tensor`, which is harvested as
    creation-not-arithmetic. Third batch running to lose a kernel to this rule
    (6, 9, 15), and the open decision about it stands unchanged.
    """
    return (
        KernelSpec(
            name="fp32_lerp", module=Fp32Lerp(),
            args=(torch.randn(256, 256), torch.randn(256, 256),
                  torch.rand(256, 256)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="linear interpolation with a TENSOR weight, and eight primitives "
                 "because of it. torch picks between two algebraically equal forms "
                 "on `w < 0.5` to keep the result stable at both ends, so the graph "
                 "contains a comparison and a `where` that the obvious one-line "
                 "formula does not. The reference is the specification: a candidate "
                 "computing a + w*(b-a) would pass the tolerance and would not be "
                 "computing the same function",
        ),
        KernelSpec(
            name="fp32_addcmul", module=Fp32Addcmul(),
            args=(torch.randn(256, 384), torch.randn(256, 384),
                  torch.randn(256, 384)),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="t + 0.5*(a*b): three input streams, one output, and the whole "
                 "kernel is two vector operations per element. It lands in T2 at a "
                 "384 KB tensor purely because there are THREE of them plus the "
                 "result -- the tier reads the working set, not the operand, and "
                 "this is the cheapest way to see the difference. At 2 ops per "
                 "element it is below the ~3-op ABLATION break-even, so the staging "
                 "grant is capacity-only and is not expected to pay",
        ),
        KernelSpec(
            name="fp32_addcdiv", module=Fp32Addcdiv(),
            args=(torch.randn(320, 256), torch.randn(320, 256),
                  torch.rand(320, 256) + 0.5),
            dtype_bytes=4, expect_tier="T2",
            expect_mechanisms=frozenset({"hvx", "l2fetch", "dma", "vtcm"}),
            note="the same shape with a DIVISION instead of a multiply, which is "
                 "the whole point of the pair: HVX has no float divide, so this one "
                 "needs a Newton reciprocal where its sibling needs one vmpy. The "
                 "divisor is drawn from [0.5, 1.5] so the reciprocal's seed is in "
                 "its accurate range and no lane is near zero -- the range is part "
                 "of the task, not a convenience",
        ),
        KernelSpec(
            name="fp16_baddbmm", module=Fp16Baddbmm(),
            args=(torch.randn(8, 64, 64, dtype=torch.float16),
                  torch.randn(8, 64, 64, dtype=torch.float16),
                  torch.randn(8, 64, 64, dtype=torch.float16)),
            dtype_bytes=2, expect_tier="T1",
            expect_mechanisms=frozenset({"hmx", "hvx", "l2fetch"}),
            note="batched matmul plus a full-rank bias -- the third tensor is the "
                 "same shape as the result, not a broadcast row like addmm's, so "
                 "there is no splat to hoist and the bias is simply the "
                 "accumulator's initial value per output tile. Six primitives: "
                 "torch decomposes the beta/alpha scaling even at the defaults. HMX "
                 "is granted through the bmm nest",
        ),
        KernelSpec(
            name="fp32_sanitize_nonfinite", module=Fp32SanitizeNonfinite(),
            args=(_with_nonfinite(torch.randn(256, 256)), torch.randn(256, 256)),
            dtype_bytes=4, expect_tier="T1",
            expect_mechanisms=frozenset({"hvx", "l2fetch"}),
            note="THE ONLY KERNEL IN THIS CORPUS WHOSE INPUT CONTAINS NaN AND BOTH "
                 "INFINITIES. Four primitives of pure predicate work -- isnan, "
                 "isinf, a bitwise or, and a three-operand where. The isnan/isinf "
                 "emitters have existed since batch 4 and every task since has "
                 "drawn its inputs from a normal distribution, so they have been "
                 "compiled 60 times and never once handed a value that reaches "
                 "them. `x != x` is the NaN test and it has to SURVIVE THE "
                 "COMPILER: -ffast-math folds it to false, which is one reason this "
                 "repo passes no such flag. isinf is `(x == x) && (x - x != 0)`, "
                 "which needs the same protection",
        ),
    )


#: The HAND-WRITTEN batches only. Named for what it holds, because the previous name
#: -- `BATCHES` -- was the defect.
#:
#: It cannot contain the mined batches 16+: those are built from
#: `run_artifacts/forge2/mined_selection.json` by `forge2.mined`, which imports
#: `KernelSpec` from this module, so importing it here at module scope is a cycle.
#: `batch()` reaches it lazily instead.
#:
#: So `for b in BATCHES` READ LIKE "for every batch" AND MEANT "for batches 1-15",
#: and that mistake was made FOUR times, every one of them passing green:
#:
#:   * the provenance compliance gate -- the corpus's central claim -- checked 75 of
#:     125 kernels, so the mined half, whose whole argument is that a machine
#:     selected them FROM the harvest, was never once asserted against the harvest;
#:   * `test_every_batch_kernel_gets_both_irs` asserted a corpus-wide invariant over
#:     batches 1 and 2, while 20 kernels violated it;
#:   * `validate_schedule`'s default scope was 15 batches of 25;
#:   * `mine`'s own "already in the corpus" filter could not see the mined kernels,
#:     so mining batch 26 would have re-selected ops already built in 16-25 -- and
#:     the duplicates would have shown up only to whoever diffed two JSON files.
#:
#: None of those was visible in the assertion. A loop over an incomplete mapping does
#: not fail, it just covers less and reports success, so the name is the only place
#: the hazard can be made visible -- hence this one. Anything that wants the WHOLE
#: corpus calls `all_batches()`.
HANDWRITTEN_BATCHES = {
    1: batch1, 2: batch2, 3: batch3, 4: batch4, 5: batch5, 6: batch6,
    7: batch7, 8: batch8, 9: batch9, 10: batch10, 11: batch11,
    12: batch12, 13: batch13, 14: batch14, 15: batch15,
}


def all_batches() -> list:
    """Every batch number in the corpus, hand-written and mined.

    `HANDWRITTEN_BATCHES` holds only 1-15, for the import-cycle reason given at its
    definition. Anything that wants to walk the WHOLE corpus calls this, and until it
    existed nothing did: `validate_schedule` with no `--batch` silently checked 15
    batches out of 25 and printed a clean summary for the two thirds it had looked
    at. An instrument that quietly narrows its own scope is worse than one that
    fails.
    """
    from hexkernels.forge import mined as _mined
    return sorted(set(HANDWRITTEN_BATCHES) | set(_mined.MINED_BATCHES))


def batch(n: int):
    """Batches 1-15 are hand-written here; 16+ are MINED.

    The mined ones live in `forge2.mined`, built from
    `run_artifacts/forge2/mined_selection.json` at import. Imported lazily so a
    missing or in-progress selection file cannot break the hand-written batches --
    and so `mined.py` can import `KernelSpec` from here without a cycle.
    """
    if n in HANDWRITTEN_BATCHES:
        return HANDWRITTEN_BATCHES[n]()
    from hexkernels.forge import mined as _mined
    if n in _mined.MINED_BATCHES:
        return _mined.batch(n)
    raise ValueError(
        f"no batch {n}; hand-written: {sorted(HANDWRITTEN_BATCHES)}, "
        f"mined: {sorted(_mined.MINED_BATCHES)}")
