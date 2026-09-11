"""Kernel specs for a Qwen3.5-0.8B vision-encoder kernel set.

Deliberately a SEPARATE module from `kernels.py`. These specs exist to produce
kernels for a downstream project, not to enter this benchmark's corpus, and
mixing them into the numbered batches would contaminate a corpus whose only
claim about itself is how it was built. Nothing here is referenced by
`kernels.batch()`, so `run_batch --batch N` can never pick them up.

SHAPES ARE MEASURED, NOT CHOSEN
-------------------------------
Every shape below was read off the actual compiled encoder graph at 256x256 --
the distinct (input shape, output shape, attrs) signatures its ops present,
counted. They are not representative sizes. The multiplicity comment on each
spec is how many ops of that exact signature the real graph contains, so the
weight each kernel carries is visible.

The op semantics are transcribed from the encoder's own reference
implementations, not inferred from op names. Two are subtle enough to be worth
stating:

  * patchify emits MERGE-BLOCK order, not raster order. The 2x2 spatial merge
    downstream is a pure reshape, so consecutive runs of merge*merge tokens
    must already BE the 2x2 blocks. Raster order would merge the wrong four
    patches and would do it silently -- every shape still checks out.

  * rope_2d rotates in float32 regardless of activation dtype, and cos/sin are
    broadcast on the HEAD axis, not the token axis.

`reshape` has no spec here on purpose: at these shapes every reshape in the
graph is row-major contiguous, so it is a pure metadata change with no bytes to
move. 49 of the graph's 396 ops need no kernel at all.

WHAT THIS PIPELINE WILL NOT PRODUCE, AND WHY
--------------------------------------------
Three of the encoder's ops are pure data movement, and the provenance gate
rejects them by design:

    every one of its N primitives is plumbing (shape, metadata or
    constant-materialising), so the kernel computes nothing and there is no
    accelerator use to measure.

That is the right call for a benchmark of accelerator use and the wrong shape
for a library, which still has to move those bytes. The two head/token
transposes (48 ops between them) and patchify (1 op) are therefore written by
hand in the consumer, not generated here. They are also the easiest kernels in
the set to verify by hand, being permutations with no arithmetic: the output is
a rearrangement of the input's exact bytes, so correctness is a permutation
check rather than a numerical one. `PLUMBING_OPS` below records them so the
omission is a documented decision and not a gap someone has to rediscover.
"""
from dataclasses import dataclass, field

import torch

from hexkernels.forge.kernels import KernelSpec

# Encoder constants, from the model's own config and code.
TOKENS = 256          # 16x16 patch grid at 256x256 with patch 16
HIDDEN = 768
HEADS = 12
HEAD_DIM = 64         # HIDDEN / HEADS
MLP = 3072
MERGED = 64           # TOKENS / (merge*merge) = 256 / 4
MERGER_IN = 3072      # HIDDEN * merge * merge
OUT_HIDDEN = 1024
LN_EPS = 1e-6         # nn.LayerNorm(hidden, eps=1e-6); NOT in config.json
ATTN_SCALE = 0.125    # 1/sqrt(64), an exact power of two


class Add(torch.nn.Module):
    """Residual add. 24 of the graph's 25 adds are this exact signature."""

    def forward(self, a, b):
        return a + b


class CastF32ToF16(torch.nn.Module):
    """The host/activation dtype boundary: an fp32 image becomes fp16 tokens."""

    def forward(self, x):
        return x.to(torch.float16)


class LayerNorm(torch.nn.Module):
    """LayerNorm over the last axis with fp32 affine params on fp16 data.

    eps is 1e-6 and is hardcoded in the model source, absent from config.json.
    The vision tower uses LayerNorm; the text tower uses RMSNorm. They are not
    interchangeable.

    WHY THE MEAN AND VARIANCE ARE SPELLED OUT rather than calling
    F.layer_norm: that decomposes to `aten.var_mean`, a fused two-output
    reduction the schedule annotator has no entry for, and tracing it fails
    outright. Writing the arithmetic explicitly also makes the emitted scalar
    reference structurally match what the kernel does -- two passes over the
    row, then a broadcast rescale -- so the reference is a fair statement of
    the problem instead of an opaque library call.

    The variance is the BIASED one (divided by N, not N-1), which is what
    LayerNorm specifies.
    """

    def forward(self, x, weight, bias):
        xf = x.to(torch.float32)
        mean = xf.mean(-1, keepdim=True)
        centred = xf - mean
        var = (centred * centred).mean(-1, keepdim=True)
        normed = centred * torch.rsqrt(var + LN_EPS)
        return (normed * weight + bias).to(x.dtype)


class BatchedMatMul(torch.nn.Module):
    """Attention's two contractions, both batched over the head axis."""

    def forward(self, a, b):
        return torch.matmul(a, b)


class Scale(torch.nn.Module):
    """Q scaling by 1/sqrt(head_dim). 0.125 is exact in fp16."""

    def forward(self, x):
        return x * ATTN_SCALE


class SoftmaxLastAxis(torch.nn.Module):
    """Attention softmax over the key axis.

    Reduce-then-broadcast twice over: a max reduction, then a sum reduction.
    Neither vectorises as a plain vector loop; the three elementwise stages
    between them do.
    """

    def forward(self, x):
        return torch.softmax(x, dim=-1)


class Permute102(torch.nn.Module):
    """[tokens, heads, head_dim] -> [heads, tokens, head_dim]. 36 ops."""

    def forward(self, x):
        return x.permute(1, 0, 2).contiguous()


class Permute021(torch.nn.Module):
    """[heads, tokens, head_dim] -> [heads, head_dim, tokens]. 12 ops."""

    def forward(self, x):
        return x.permute(0, 2, 1).contiguous()


class LinearBias(torch.nn.Module):
    """A[M,K] @ W[K,N] + bias[N], no activation.

    W arrives already transposed from PyTorch's Linear.weight [out, in]; that
    transpose happens once at graph-build time, so no runtime transpose op
    exists for any projection.
    """

    def forward(self, a, w, bias):
        return a @ w + bias


class LinearBiasGeluTanh(torch.nn.Module):
    """The MLP's first layer: matmul + bias + tanh-approximate GELU, fused.

    hidden_act is gelu_pytorch_tanh, so this is the tanh approximation and not
    exact erf. The two are different kernels and the encoder uses both.
    """

    def forward(self, a, w, bias):
        return torch.nn.functional.gelu(a @ w + bias, approximate="tanh")


class LinearBiasGeluErf(torch.nn.Module):
    """The patch merger's first layer: matmul + bias + EXACT-erf GELU.

    Exact erf, not the tanh approximation: the merger constructs a plain
    nn.GELU() with no `approximate` argument, unlike the blocks' MLP. Used once,
    but it holds the largest weight tensor in the encoder.
    """

    def forward(self, a, w, bias):
        return torch.nn.functional.gelu(a @ w + bias, approximate="none")


class Rope2D(torch.nn.Module):
    """2D rotary embedding applied to Q and K in every block.

    The rotation is computed in float32 regardless of the activation dtype and
    cast back at the end. cos/sin are [tokens, head_dim] and broadcast across
    the HEAD axis.
    """

    def forward(self, x, cos, sin):
        xf = x.to(torch.float32)
        cosf = cos.to(torch.float32).unsqueeze(1)
        sinf = sin.to(torch.float32).unsqueeze(1)
        half = HEAD_DIM // 2
        rotated = torch.cat((-xf[..., half:], xf[..., :half]), dim=-1)
        return (xf * cosf + rotated * sinf).to(x.dtype)


class Patchify(torch.nn.Module):
    """[C,T,H,W] image -> [tokens, C*T*patch*patch] in MERGE-BLOCK order.

    Token axes come out ordered (block_h, block_w, merge_h, merge_w) and feature
    axes (channel, T, patch_h, patch_w). The feature order matches the Conv3d
    weight layout [embed, C, T, ph, pw] it is multiplied against.
    """

    def forward(self, img):
        c, t, h, w = 3, 2, 256, 256
        patch, merge = 16, 2
        grid_h, grid_w = h // patch, w // patch
        x = img.reshape(c, t, grid_h, patch, grid_w, patch)
        x = x.reshape(c, t, grid_h // merge, merge, patch,
                      grid_w // merge, merge, patch)
        x = x.permute(2, 5, 3, 6, 0, 1, 4, 7)
        return x.reshape(grid_h * grid_w, c * t * patch * patch)


def _f16(*shape):
    return torch.randn(*shape, dtype=torch.float16)


def _f32(*shape):
    return torch.randn(*shape, dtype=torch.float32)


# Ops this pipeline cannot produce, with the multiplicity each carries in the
# real graph. Hand-written in the consumer; see the module docstring.
PLUMBING_OPS = (
    ("transpose_tokens_heads", 36, "[tokens,heads,dim] -> [heads,tokens,dim]"),
    ("transpose_keys", 12, "[heads,tokens,dim] -> [heads,dim,tokens]"),
    ("patchify", 1, "[C,T,H,W] -> [tokens, C*T*patch*patch], merge-block order"),
)


def encoder_batch():
    """The encoder's generatable kernel set: 10 specs, 298 of the graph's 396 ops.

    The other 98: 49 reshapes that move no bytes, and 49 pure-plumbing ops
    (`PLUMBING_OPS`) the provenance gate rejects and the consumer hand-writes.
    """
    return (
        # ---- elementwise and layout: cheap per op, but 158 ops of them ----
        KernelSpec(
            name="enc_add_fp16", module=Add(),
            args=(_f16(TOKENS, HIDDEN), _f16(TOKENS, HIDDEN)),
            dtype_bytes=2, expect_tier="T2",
            note="24 ops. Residual add, unit stride, one vector loop. The "
                 "25th add takes an fp32 right operand (the learned pos_embed) "
                 "and is handled by casting rather than by a second kernel",
        ),
        KernelSpec(
            name="enc_scale_fp16", module=Scale(),
            args=(_f16(HEADS, TOKENS, HEAD_DIM),),
            dtype_bytes=2, expect_tier="T1",
            note="12 ops. Multiply by 1/sqrt(64)=0.125, which is a power of "
                 "two and therefore exact in fp16 -- no rounding to budget for",
        ),
        KernelSpec(
            name="enc_cast_f32_f16", module=CastF32ToF16(),
            args=(_f32(TOKENS, MERGER_IN // 2),),
            dtype_bytes=4, expect_tier="T2",
            note="1 op. The host/activation dtype boundary, made explicit in "
                 "the IR so no other op can smuggle a dtype change",
        ),
        # ---- normalisation and softmax: reduce-then-broadcast ----
        KernelSpec(
            name="enc_layernorm_fp16", module=LayerNorm(),
            args=(_f16(TOKENS, HIDDEN), _f32(HIDDEN), _f32(HIDDEN)),
            dtype_bytes=2, expect_tier="T2",
            note="25 ops. Two reductions (mean, variance) then a broadcast "
                 "rescale, with fp32 affine params on fp16 data. eps 1e-6",
        ),
        KernelSpec(
            name="enc_softmax_fp16", module=SoftmaxLastAxis(),
            args=(_f16(HEADS, TOKENS, TOKENS),),
            dtype_bytes=2, expect_tier="T2",
            note="12 ops. Max reduction, exp, sum reduction, reciprocal "
                 "broadcast. The exponential must be computed on the fp32 "
                 "path: the obvious fp16 exp2 helper in the reference library "
                 "has a wrong polynomial coefficient and is unusable above "
                 "|frac| 0.25, which is exactly this kernel's input range",
        ),

        # ---- rotary embedding ----
        KernelSpec(
            name="enc_rope2d_fp16", module=Rope2D(),
            args=(_f16(TOKENS, HEADS, HEAD_DIM), _f32(TOKENS, HEAD_DIM),
                  _f32(TOKENS, HEAD_DIM)),
            dtype_bytes=2, expect_tier="T2",
            note="24 ops. Rotate-half then fused multiply-add against a cos/sin "
                 "table broadcast over heads. The half-rotation is a shuffle "
                 "with a sign flip on one half, not an arithmetic op",
        ),

        # ---- contractions: where the cycles actually are ----
        KernelSpec(
            name="enc_matmul_qk_fp16", module=BatchedMatMul(),
            args=(_f16(HEADS, TOKENS, HEAD_DIM), _f16(HEADS, HEAD_DIM, TOKENS)),
            dtype_bytes=2, expect_tier="T2",
            note="12 ops. Q@K^T, batched over heads. fp16 contraction with a "
                 "short K=64 reduction, so HMX tiles apply but the reduction "
                 "strip is only two tiles deep",
        ),
        KernelSpec(
            name="enc_matmul_av_fp16", module=BatchedMatMul(),
            args=(_f16(HEADS, TOKENS, TOKENS), _f16(HEADS, TOKENS, HEAD_DIM)),
            dtype_bytes=2, expect_tier="T2",
            note="12 ops. attn@V, batched over heads. K=256 here, so the "
                 "reduction is eight tiles deep and the accumulator lives "
                 "far longer than in Q@K^T",
        ),
        KernelSpec(
            name="enc_linear_bias_fp16", module=LinearBias(),
            args=(_f16(TOKENS, HIDDEN), _f16(HIDDEN, HIDDEN), _f32(HIDDEN)),
            dtype_bytes=2, expect_tier="T2",
            note="48 ops -- qkv and proj in every block, the most numerous "
                 "contraction. Bias is added once before the tile loops, not "
                 "per output tile",
        ),
        KernelSpec(
            name="enc_linear_gelu_tanh_fp16", module=LinearBiasGeluTanh(),
            args=(_f16(TOKENS, HIDDEN), _f16(HIDDEN, MLP), _f32(MLP)),
            dtype_bytes=2, expect_tier="T2",
            note="12 ops. The MLP's expanding layer with a fused "
                 "tanh-approximate GELU. Fusing the activation into the "
                 "matmul epilogue is what keeps the [256,3072] intermediate "
                 "from being written and re-read",
        ),
        KernelSpec(
            name="enc_linear_gelu_erf_fp16", module=LinearBiasGeluErf(),
            args=(_f16(MERGED, MERGER_IN), _f16(MERGER_IN, MERGER_IN),
                  _f32(MERGER_IN)),
            dtype_bytes=2, expect_tier="T3",
            note="1 op, and it holds the largest weight tensor in the whole "
                 "encoder -- 3072x3072, larger than any transformer layer. "
                 "EXACT-erf GELU, not the tanh approximation the MLP uses. "
                 "The weight alone exceeds VTCM, so this one must stream",
        ),
    )
