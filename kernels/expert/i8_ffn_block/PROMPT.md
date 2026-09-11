# int8 transformer FFN / MLP block — 2 matmuls + gated activation (HMX x2 + VTCM + HVX)

Implement
`candidate_kernel(const uint8_t *X, const int8_t *W1, const int32_t *b1, const int8_t *W2, const int32_t *b2, int8_t *out, int S, int D, int Dff)`
computing the transformer feed-forward layer: an up-projection matmul, a ReLU +
requant activation, and a down-projection matmul, all in int8. `S == 64`,
`D == Dff == 128`:

```
acc1[i][j] = sum_{k<D}  X[i*D+k] * W1[k*Dff+j]          (X uint8 0..3, W1 int8 -3..3)   i<S, j<Dff
r1         = sign_extend_12bit((acc1*17 + 8) >> 4)       (0x40-config HMX requant field)
H[i][j]    = (r1+b1[j] > 0) ? clamp((r1+b1[j]) >> 8, 0, 127) : 0   (ReLU + requant -> int8)

acc2[i][j] = sum_{k<Dff} H[i*Dff+k] * W2[k*D+j]         (W2 int8 -2..2)                 i<S, j<D
r2         = sign_extend_12bit((acc2*17 + 8) >> 4)
out[i][j]  = saturate_int8((r2 + b2[j]) >> 3)            (int8 output logits)
```

This is the canonical on-device transformer FFN: it must COMPOSE three mechanism
groups.

- **HMX** for BOTH matmuls, each over a 2x4 grid of 32x32 output tiles accumulating
  over 4 K-tiles. Clear the accumulator once per output tile (`mxclracc`), issue the
  four K-tile `(activation, weight)` matmul packets, then the `bias=mxmem` +
  `mxmem(...):after.uh=acc:2x1` requant store.
- **VTCM** for the intermediate activation H: keep H on-chip between the two matmuls
  (bulk-copy it into VTCM after matmul1, bulk-copy it back out to pack matmul2's
  activation croutons) alongside the HMX crouton operand slots. The harness installed
  the identity VTCM `add_translation` and enabled the HMX context before calling you.
- **HVX** for the on-chip data movement (bulk 128B vector copies of the crouton
  operands / output / config) and the fused ReLU/requant/bias/saturate epilogues.

Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte
of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B
tile, load limit 1023) — activation and weight need DIFFERENT load limits in one
packet; output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`. Stage crouton
pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B vector copies
(a scalar VTCM access costs ~48 cyc in timing mode and would lose to the baseline).

Input ranges keep `|r1|, |r2| < 2048` so both 12-bit requant fields are exact (no HMX
saturation) and the output is bit-exact int8. A competent HVX `vrmpy` FFN (both
matmuls as dot-products, H in plain DDR, no HMX) is the baseline; the composed expert
must beat it by >=1.2x.


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
