# int8 grouped-query attention BLOCK (QK^T-HMX -> softmax-HVX -> A.V-HMX, shared-KV-head reuse)

Implement `candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V, uint16_t *out, const uint8_t *exp_lut)`
computing a full grouped-query attention layer where `H_Q=2` query heads SHARE `H_KV=1`
key/value head (`GROUP_SIZE=2`), over `S=64` tokens of head-dim `D=64`. `Q` is uint8
(0..7), `K`,`V` are int8 (-3..3). `Q` is `[H_Q,S,D]`, `K`/`V` are `[H_KV,S,D]`, `out` is
`[H_Q,S,D]` uint16, all head-outermost row-major.

Per query head `h_q` (shared KV head `h_kv = h_q / GROUP_SIZE`):

```
scores[i][j] = sum_d Q[h_q,i,d] * K[h_kv,j,d]        (Q.K^T; K row j IS key vector j)
s12[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF          (0x40-config HMX requant field)
scaled[i][j] = clamp( sign_extend_12bit(s12) >> 4 , -128, 127)   (the attention SCALE)
probs[i][:]  = softmax_lut( scaled[i][:] ) over the KEY axis j    (uint8, row sum ~255)
                 m    = max_j scaled[i][j]
                 e_j  = exp_lut[ clamp(scaled[i][j]-m, -255, 0) + 255 ]
                 probs[i][j] = (e_j*255 + (sum_j e_j)/2) / (sum_j e_j)
out_raw[i][d]= sum_j probs[i][j] * V[h_kv,j,d]                    (probs uint8, V int8)
out[h_q,i,d] = ((out_raw*17 + 8) >> 4) & 0xFFF                    (0x40-config HMX requant)
```

This is an L3 GQA transformer layer that must COMPOSE three mechanism groups AND exploit
the grouped-query structure.

- **HMX** runs both matmuls (QK^T and A.V) on the matrix engine, each a 2x2 grid of 32x32
  crouton output tiles accumulating over 2 reduction tiles. `mxclracc` once per output
  tile, issue the reduction-tile `(activation, weight)` matmul packets, then the
  0x40-config requant store. QK^T packs the weight from `K` stored `[S,D]` with swapped
  tile indices; A.V packs `V` as a generic `B[K,N]` operand (no transpose), reducing over
  the key axis `S`.
- **GQA reuse (the mechanism win):** because `H_Q` query heads share one KV head, pack the
  shared KV head's K and V weight croutons into VTCM ONCE, up front, and leave them
  resident -- both query heads reuse them without re-packing. Only the query (QK^T) and
  probs (A.V) activation side is re-packed per head. This removes the expensive
  weight-crouton pack + bulk-copy for all but the first head.
- **HVX** runs the scale between the two matmuls in vector lanes (`vasl` by 4 then
 `vasr` by 4 + 4 sign-extends the 12-bit requant field and applies the >>4 in one
 pair, then `vpack:sat` clamps to int8 in hardware). The row-wise softmax after it is
 necessarily SCALAR -- the exp LUT is far larger than the 32-byte `vlut32` gather and
 the normalise is an integer divide by a per-row scalar that must stay bit-exact.
 HVX also does the bulk 128B
  vector copies that stage crouton tiles to/from VTCM.
- **VTCM** holds the HMX operand/output crouton tiles (`HVX_VTCM_BASE`), including the
  resident shared-KV weight croutons. Crouton pack/unpack is staged in cacheable DDR and
  moved to/from VTCM in bulk vector copies (scalar VTCM access costs ~48 cyc in timing
  mode). The scores/probs intermediates live in cacheable DDR between stages (at 4KB far
  too small for DMA to beat a vector copy). The harness enables the HMX context before
  calling you.

Crouton layouts (helpers in `harness_common.h`): activation = uint8 in the HIGH byte of an
fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed int8 (1024B tile,
load limit 1023) — activation and weight need DIFFERENT load limits in one packet; output =
crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.

The whole block is fixed-point integer, so the LUT softmax makes it BIT-EXACT to the scalar
reference (no tolerance). Input ranges keep every requant field `< 2048`. A competent HVX
`vrmpy` implementation of the whole GQA block (both matmuls + softmax per head, no HMX, no
weight reuse) is the baseline; the composed expert must beat it by >=1.2x.


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
