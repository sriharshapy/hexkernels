# fp16 32x32x128 (deep-K) matmul + leaky ReLU on the HMX matrix engine

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out,
int n, int k_dim)` for `n == 32`, `k_dim == 128`:

```
acc[i][j] = sum_k A[i*k_dim+k] * B[k*n+j]     (float32-accumulate, M=N=32, K=128)
m[i][j]   = (__fp16)acc[i][j]                  (fp16-round: matches HMX output)
v         = (float)m[i][j]
out[i][j] = v > 0 ? v : v * 0.125f             (leaky ReLU, slope 1/8)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- Output is compared to the float32-accumulate scalar reference with an fp16
  tolerance -- HVX/HMX float arithmetic is non-IEEE (qf16).
- K=128 is 4x deeper than the single-tile sibling. Clear the HMX float
  accumulator ONCE (`Q6_mxclracc_hf`), then issue all four (activation, weight)
  load-matmul pairs before the single store.
- Fuse leaky ReLU into the epilogue as native HVX vector ops on the
  crouton-order output block: `pos=max(v,0)`, `neg=min(v,0)` (both native
  `Vhf` ops at v68), `neg*0.125` (native qf16 multiply), then
  `pos+scaled` (qf16 add -- v68 has no native `Vhf+Vhf` add, needs v79+) and
  convert once. Do NOT round-trip through scalar `(float)` casts per element.
- A competent HVX qf16 row-broadcast matmul (over K=128) with the identical
  leaky-ReLU epilogue is the baseline; the HMX expert must beat it by >=1.2x
  (measured: 1.23x).


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
