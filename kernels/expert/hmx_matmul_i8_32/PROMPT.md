# int8 32x32 matmul on the HMX matrix engine (output int32 requant field)

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n)`
for `n == 32` (square, K == n):

```
acc[i][j] = sum_{k=0}^{n-1} A[i*n+k] * B[k*n+j]      (A uint8 0..7, B int8 -3..3)
out[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 )    (0x40-config HMX requant field)
```

- The harness has enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- Pack A into the activation crouton (int8 in HIGH byte, 2048B, load limit 2047)
  and B into the 4-deep weight tile (1024B, load limit 1023); `mxclracc`; issue a
  `{ activation.ub=mxmem(...); weight.b=mxmem(...) }` matmul packet (the two
  operands need DIFFERENT load limits); load the `0x40` requant config
  (`bias=mxmem`); store `mxmem(...):after.uh=acc:2x1`; then unpack, sign-extend
  the 12-bit field, and write int32.
- Stage crouton pack/unpack through cacheable buffers + bulk 128B vector copies
  (scalar VTCM access is ~48 cyc each in timing mode). The HMX expert must beat a
  competent HVX `vrmpy` matmul by >=1.2x.
- Layout helpers in `harness_common.h`: `hvx_hmx_i8_act_off`, `hvx_hmx_i8_wgt_off`,
  `hvx_hmx_i8_out_off`.

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
