# int8 32x32 HMX deep-K GEMM (K = 128) fused with ReLU

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out, int n, int K)`
for `n == 32`, `K == 128`:

```
acc[i][j] = sum_{k=0}^{K-1} A[i*K+k] * B[k*n+j] (A uint8 0..3, B int8 -1..1)
r[i][j] = sign_extend_12bit( (acc*17 + 8) >> 4 ) (0x40-config HMX requant field)
out[i][j] = r[i][j] > 0 ? r[i][j] : 0 (ReLU)
```

- The output is a single 32x32 tile, but the contraction K = 128 = 4*32, so the
 matmul accumulates over FOUR K-tiles. Issue ONE `mxclracc`, then for each of the
 `K/32` contraction tiles pack the `(activation, weight)` 32x32 pair into VTCM
 and issue one accumulating `{ activation.ub=mxmem(...); weight.b=mxmem(...) }`
 packet. After all four, load the `0x40` requant config, do ONE requant store,
 unpack, sign-extend the 12-bit field, and clamp negatives to 0 (ReLU).
- A is row-major `[n][K]`, B is row-major `[K][n]`, out is `[n][n]`.
- The long running accumulation is why the input range is narrow (keeps the
 12-bit requant field exact). Do NOT drop or re-clear tiles.
- The ReLU is fused into the HVX unpack epilogue, not a separate pass.
- Stage crouton pack/unpack through cacheable buffers + bulk 128B vector copies;
 beat a competent scalar deep-K matmul+ReLU by >=1.2x.
- Layout helpers: `hvx_hmx_i8_act_off`, `hvx_hmx_i8_wgt_off`, `hvx_hmx_i8_out_off`.

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
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
