# FP16 32x32 matmul + GELU epilogue (HMX)

Implement `candidate_kernel(const __fp16 *A, const __fp16 *B, __fp16 *out, int n)`
computing `out[i*n+j] = gelu( sum_k A[i*n+k]*B[k*n+j] )` for `n == 32`.

- Use the Hexagon HMX engine for the matrix multiply (crouton layout).
- After the HMX matmul, apply the **pinned tanh GELU**:
  `gelu(x) = 0.5*x*(1 + tanhf(0.7978845608*(x + 0.044715*x*x*x)))` in float.
- Apply GELU to the **fp16-rounded HMX output** (read from the crouton unpack),
  then cast the result back to `__fp16`.
- The harness enables the HMX context before calling you; use VTCM at `HVX_VTCM_BASE`.
- Output must be bit-exact to the scalar reference (float-accum → cast to __fp16 →
  gelu_f32 → cast to __fp16).


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
