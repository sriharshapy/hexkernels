# int8 3x3 convolution (VALID) via im2col -> HMX matrix engine

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n)`
computing a 3x3 VALID convolution with the HMX requant epilogue, for the fixed shape
`C_in=8`, `C_out=32`, input `10x10`, output `8x8` (`P=64` positions):

```
acc[co][oh][ow] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci][kh][kw]   (in u8, W i8)
out[co*P + p]   = ((acc*17 + 8) >> 4) & 0xFFF          (0x40-config requant, 12-bit field)
```

- Lower the conv to a matmul: rows = output positions (M=P), reduction = the
  `(ci,kh,kw)` receptive field (K = C_in*Kh*Kw = 72), cols = output channels (N=C_out).
- **im2col**: gather each output position's 3x3xC_in receptive field into a crouton
  activation row while packing. K=72 is not a multiple of 32 — zero-pad up to 3 crouton
  K-tiles (the staging is zero-initialised; out-of-range k slots stay 0).
- The harness has already enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- Keep the pack division-free: precompute `koff[k]` (= ci*IH*IW + kh*IW + kw) and
  `poff[p]` (= oh*IW + ow) so the input index is `koff[kg] + poff[p]`; with the k-index
  ordered `kg = ci*Kh*Kw + kh*Kw + kw` the weight value is simply `W[co*K + kg]`.
- Clear the accumulator once per output tile (`mxclracc`), issue the 3 K-tile matmul
  packets, then the 0x40 requant store; stage crouton pack/unpack via cacheable buffers +
  bulk 128B vector copies.
- A competent HVX im2col+`vrmpy` conv is the baseline; the HMX expert must beat it by >=1.2x.


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
