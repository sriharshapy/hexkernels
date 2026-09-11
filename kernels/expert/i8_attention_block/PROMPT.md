# int8 single-head attention BLOCK (QK^T-HMX -> softmax-HVX -> A.V-HMX, VTCM-staged)

Implement `candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V, uint16_t *out, int S, int D, const uint8_t *exp_lut)`
computing a full single-head attention layer over `S=64` tokens of head-dim `D=64`.
`Q` is uint8 (0..7), `K` and `V` are int8 (-3..3):

```
scores[i][j] = sum_d Q[i*D+d] * K[j*D+d] (Q.K^T; K row j IS key vector j)
s12[i][j] = ((scores*17 + 8) >> 4) & 0xFFF (0x40-config HMX requant field)
scaled[i][j] = clamp( sign_extend_12bit(s12) >> 4 , -128, 127) (the attention SCALE)
probs[i][:] = softmax_lut( scaled[i][:] ) over the KEY axis j (uint8, row sum ~255)
 m = max_j scaled[i][j]
 e_j = exp_lut[ clamp(scaled[i][j]-m, -255, 0) + 255 ]
 probs[i][j] = (e_j*255 + (sum_j e_j)/2) / (sum_j e_j)
out_raw[i][d]= sum_j probs[i][j] * V[j*D+d] (probs uint8, V int8)
out[i*D+d] = ((out_raw*17 + 8) >> 4) & 0xFFF (0x40-config HMX requant)
```

This is the marquee L3 kernel: a real transformer attention layer that must COMPOSE
three mechanism groups.

- **HMX** runs both matmuls (QK^T and A.V) on the matrix engine, each a 2x2 grid of
 32x32 crouton output tiles accumulating over 2 reduction tiles. `mxclracc` once per
 output tile, issue the reduction-tile `(activation, weight)` matmul packets, then the
 0x40-config requant store. QK^T packs the weight from `K` stored `[S,D]` with swapped
 tile indices (`K` is already "pre-transposed" for scores = Q.K^T); A.V packs `V` as a
 generic `B[K,N]` operand (no transpose), reducing over the key axis `S`.
- **HVX** runs the scale between the two matmuls in vector lanes — sign-extend the
 12-bit requant field and apply the `>>4` in halfword lanes (one `vasl` by 4 then one
 `vasr` by 8 does both), then `vpack:sat` down to int8, which performs the clamp in
 hardware. It also does the bulk 128B vector copies that stage crouton tiles to/from
 VTCM. The softmax that follows is necessarily SCALAR and you should leave it so: the
 exp LUT has 511 entries, far past the 32-byte `vlut32` gather, and the normalise is an
 integer divide by a per-row scalar that must stay bit-exact.
- **VTCM** holds the HMX operand/output crouton tiles (`HVX_VTCM_BASE`). Crouton
 pack/unpack is staged in cacheable DDR and moved to/from VTCM in bulk vector copies
 (scalar VTCM access costs ~48 cyc in timing mode). The scores/probs intermediates
 live in cacheable DDR between stages (at 4KB they are far too small for DMA to beat a
 vector copy). The harness enables the HMX context before calling you.

Crouton layouts (helpers in `harness_common.h`): activation = uint8 in the HIGH byte of
an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed int8 (1024B
tile, load limit 1023) — activation and weight need DIFFERENT load limits in one packet;
output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.

The whole block is fixed-point integer, so the LUT softmax makes it BIT-EXACT to the
scalar reference (no tolerance). Input ranges keep every requant field `< 2048` (probs
is a normalized row summing to ~255, so the A.V accumulator stays bounded). A competent
HVX `vrmpy` implementation of the whole block (both matmuls + softmax, no HMX) is the
baseline; the composed expert must beat it by >=1.2x.


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
