# int8 3x3 conv (VALID) + fused bias + ReLU on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
int32_t *out, int n)` computing a 3x3 VALID convolution with a fused per-output-channel
bias + ReLU epilogue, for the fixed shape `C_in=8`, `C_out=32`, input `10x10`, output
`8x8` (`P=64`):

```
acc[co][oh][ow] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci][kh][kw]   (in u8, W i8)
r   = sign_extend_12( (acc*17 + 8) >> 4 )              (native HMX 0x40 requant)
out[co*P + p] = max(r + bias[co], 0)                   (fused bias + ReLU)
```

- Lower the conv to a matmul via **im2col** (M=P, K=C_in*Kh*Kw=72, N=C_out), exactly as
  `i8_conv2d_3x3_hmx`; K=72 is zero-padded to 3 crouton K-tiles.
- The harness has already enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- The HMX 0x40-config store gives the native requant field; sign-extend it from 12 bits,
  add the per-output-channel int32 `bias[co]`, then ReLU-clamp to `>= 0` — fuse this into
  the HVX unpack loop.
- Keep the pack division-free (precompute `koff[k]`/`poff[p]`; weight value = `W[co*K + kg]`),
  clear the accumulator once per output tile, issue the 3 K-tile matmul packets, and stage
  crouton pack/unpack via cacheable buffers + bulk 128B vector copies.
- A competent HVX im2col+`vrmpy` conv with the same epilogue is the baseline; the HMX expert
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
