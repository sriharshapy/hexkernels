# int8 transformer ENCODER BLOCK (attention + FFN + 2 residuals, HMX x4 + VTCM + HVX)

Implement
`candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1, const int8_t *W2, const int32_t *b2, const uint8_t *exp_lut, int8_t *out, int S, int D, int Dff)`
computing a full transformer encoder block over `S=64` tokens of model-dim `D=64` with
FFN hidden `Dff=128`. `X` is uint8 (0..3, used directly as self-attention Q,K,V),
`W1` int8 (-3..3), `W2` int8 (-2..2):

```
--- Self-attention sub-layer (Q = K = V = X) ---
scores[i][j] = sum_d X[i*D+d] * X[j*D+d] (X.X^T)
scaled[i][j] = clamp( sx12((scores*17+8)>>4) >> ATTN_SCALE_SHIFT , -128,127)
probs[i][:] = softmax_lut(scaled[i][:]) over the KEY axis j (uint8, sum ~255)
av[i][d] = sum_j probs[i][j] * X[j*D+d]
a[i][d] = clamp( sx12((av*17+8)>>4) >> ATTN_OUT_SHIFT , -128,127)

--- Residual 1 ---
h[i][d] = clamp_u8( X[i*D+d] + a[i*D+d] ) (uint8, FFN activation)

--- FFN sub-layer ---
acc1[i][j] = sum_k h[i*D+k] * W1[k*Dff+j]
H[i][j] = (sx12((acc1*17+8)>>4)+b1[j] > 0) ? clamp((...)>>FFN_SH1, 0,127) : 0
acc2[i][j] = sum_k H[i*Dff+k] * W2[k*D+j]
f[i][j] = sat_i8( (sx12((acc2*17+8)>>4) + b2[j]) >> FFN_SH2 )

--- Residual 2 ---
out[i][d] = sat_i8( h[i*D+d] + f[i*D+d] ) (int8 block output)
```
(`ATTN_SCALE_SHIFT=4, ATTN_OUT_SHIFT=8, FFN_SH1=8, FFN_SH2=4`.)

This is the marquee L3 composite: it COMPOSES the attention-block and FFN-block
pipelines with two residual adds, over three mechanism groups.

- **HMX** runs ALL FOUR matmuls on the matrix engine (X.X^T, A.V, up-proj, down-proj),
 each a grid of 32x32 crouton output tiles accumulating over the reduction tiles
 (`mxclracc` once per output tile, then the 0x40-config requant store).
- **VTCM** holds the HMX operand/output crouton tiles AND the FFN intermediate H, kept
 on-chip between the two FFN matmuls (bulk-copied in after the up-proj, bulk-copied
 back out to pack the down-proj activation). The harness enables the HMX context and
 installs the identity VTCM translation before calling you.
- **HVX** runs the attention scale in vector lanes -- sign-extend the 12-bit requant
 field and apply the shift together (one `vasl` by 4, then one `vasr` by 4 + the
 scale shift), then `vpack:sat` down to int8 so the clamp is the hardware's. The
 row-wise softmax after it is necessarily SCALAR and should stay so: the exp LUT is
 far larger than the 32-byte `vlut32` gather and the normalise is an integer divide
 by a per-row scalar that must stay bit-exact. HVX also runs the residual adds and
 requant epilogues,
 and the bulk 128B vector copies that stage crouton tiles / H to/from VTCM (a scalar
 VTCM access costs ~48 cyc in timing mode and would lose).

Documented simplifications (fit the bare-sim budget + the 12-bit HMX requant field
chained across four matmuls): self-attention uses X directly as Q,K,V (no separate
QKV/output projections); LayerNorm is omitted in favour of a residual+requant path.
Input ranges + the shift constants keep every requant field `< 2048`, so the whole
block is BIT-EXACT int8. Crouton layouts (helpers in `harness_common.h`): activation =
int8 in the HIGH byte of an fp16-crouton slot (2048B tile, lim 2047); weight = 4-deep
packed (1024B tile, lim 1023) — different load limits in one packet; output = uint16
`mxmem(...):after.uh=acc:2x1`. A competent HVX `vrmpy` implementation of the whole
block (four matmuls + softmax + residuals, no HMX) is the baseline; the composed expert
must beat it by >=1.2x (measured 1.73x). Do NOT hardcode the exp_lut — it is a runtime input.


## Available helper (HMX matrix engine)

You MAY `#include "hmx_helpers.h"` and call these sim-verified helpers instead of hand-writing
the crouton `mxmem` sequence. Each owns the crouton pack + `mxclracc`/`mxmem` matmul + VTCM
staging for `M,N,K` all multiples of 32; you do the epilogue (bias/relu/residual/requant/cast)
yourself in HVX or scalar.

int8 (A uint8 row-major MxK, B int8 row-major KxN):
```
void hmx_tile_matmul_i8(const uint8_t *A, const int8_t *B, int32_t *C, int M, int N, int K);
    /* C = sx12((sum_k A*B)*17+8 >> 4): the HMX 0x40-config 12-bit requant field, sign-extended */
void hmx_tile_matmul_i8_field(const uint8_t *A, const int8_t *B, uint16_t *field, int M, int N, int K);
    /* the raw uint16 crouton field, before sign-extension */
```
fp16 (tolerance comparison, not bit-exact):
```
void hmx_tile_matmul_fp16(const __fp16 *A, const __fp16 *B, float *C, int M, int N, int K);
void hmx_tile_matmul_fp16_field(const __fp16 *A, const __fp16 *B, __fp16 *field, int M, int N, int K);
```
Using the helper is optional — a correct kernel by any means is accepted.
