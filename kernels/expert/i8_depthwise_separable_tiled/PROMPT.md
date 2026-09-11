# Tiled int8 depthwise-separable conv block (HVX depthwise + HMX pointwise + DMA/VTCM)

Implement `candidate_kernel(const uint8_t *in, const int8_t *Wdw, const int8_t *Wpw,
const int32_t *bias, uint8_t *out, int n)` — a depthwise-separable convolution block:
a 3x3 VALID **depthwise** (per-channel) followed by a 1x1 **pointwise** (dense matmul)
with a fused per-output-channel bias + ReLU + saturating requant-to-uint8 epilogue.
NHWC layout, fixed shape `C_in=64`, `C_out=64`, input `18x18`, output `16x16` (`P=256`):

```
dw[p][c]   = saturate_u8( max( sum_{kh,kw} in[oh+kh][ow+kw][c] * Wdw[c][kh][kw], 0 ) )   (per-channel)
acc[p][co] = sum_c dw[p][c] * Wpw[co][c]                                                  (1x1 = matmul)
r          = sign_extend_12( (acc*17 + 8) >> 4 )                (native HMX 0x40 requant)
out[p][co] = saturate_u8( max(r + bias[co], 0) )               (fused bias + ReLU + narrow)
```

Compose THREE mechanism groups:

- **HVX depthwise.** The 3x3 depthwise is per-channel and does NOT lower to a dense
  matmul — keep it on HVX. In NHWC the C_in=64 channels are innermost, so vectorize
  across channels (64 int16 lanes = one HVX vector): widen the input bytes IN ORDER with
  `Q6_Wh_vunpack_Vb` (NOT `vzxt`, which deinterleaves even/odd lanes), multiply by the
  per-channel per-tap weight vector, accumulate 9 taps, ReLU.
- **HMX pointwise.** The 1x1 conv is a dense matmul (M=P=256, K=C_in=64, N=C_out=64) in
  32x32 crouton tiles. Pack the pointwise weight croutons + uDMA them DDR->VTCM ONCE
  (resident, reused across all M-tiles). Pack each M-tile's dw activation crouton once and
  DOUBLE-BUFFER it: while HMX computes M-tile ti's whole tj x kt grid from VTCM buffer
  `ti&1`, uDMA-prefetch M-tile ti+1's croutons into the alternate buffer. The Type-0 DMA
  descriptor must be static/global; chain the K-tiles via the next-ptr.
- **HVX fused epilogue.** Bulk-copy the HMX output crouton VTCM->cacheable, sign-extend
  the 12-bit requant field, add `bias[co]`, ReLU, saturate to uint8.

The harness has enabled the HMX context and installed an identity VTCM translation; use
VTCM scratch at `HVX_VTCM_BASE`. A competent all-HVX block (HVX depthwise + HVX vrmpy
pointwise, same epilogue) is the baseline; the HMX+DMA expert must beat it by >=1.2x.


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
