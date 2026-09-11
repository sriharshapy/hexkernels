# int8 32x32 matmul with activation zero-point on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int zp, int32_t
*out, int n)` for `n == 32`:

```
acc[i][j] = sum_k (A[i*n+k] - zp) * B[k*n+j]
out[i][j] = sign_extend_12bit((acc*17 + 8) >> 4)
```

`A` is uint8 (0..7), `zp` is a small nonzero activation zero-point, `B` is int8
(-3..3). The `(acc*17+8)>>4 & 0xFFF` requant is the HMX matrix engine's native
output at the `0x40` bias-config.

The zero-point distributes:
`sum_k (A-zp)*B = sum_k A*B - zp * sum_k B[k][j]`, so realize it on HMX as TWO
matmuls that ACCUMULATE (pre-requant) into one accumulator: matmul `(A, B)` then
matmul `(constant-zp activation, -B weight)`. Clear the accumulator ONCE
(`mxclracc`), issue both matmul packets, then a single requant store.

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE`. int8 activation in the HIGH byte of a crouton slot
  (`hvx_hmx_i8_act_off`), weight 4-deep packed (`hvx_hmx_i8_wgt_off`), output
  crouton read as uint16 (`hvx_hmx_i8_out_off`).
- Stage crouton pack/unpack through cacheable buffers + bulk 128B copies; the HMX
  expert must beat the HVX vrmpy baseline by >=1.2x.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
code block.


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
