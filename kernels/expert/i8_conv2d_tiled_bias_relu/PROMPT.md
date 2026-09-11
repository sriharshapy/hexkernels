# Tiled int8 3x3 conv (VALID) + fused bias + ReLU on HMX with DMA/VTCM double-buffering

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
uint8_t *out, int n)` — an int8 3x3 VALID convolution with a fused per-output-channel
bias + ReLU + saturating requant-to-uint8 epilogue, for the fixed shape `C_in=16`,
`C_out=64`, input `18x18`, output `16x16` (`P=256`):

```
acc[co][p] = sum_{ci,kh,kw} in[ci][oh+kh][ow+kw] * W[co][ci*Kh*Kw + kh*Kw + kw]   (in u8, W i8)
r          = sign_extend_12( (acc*17 + 8) >> 4 )        (native HMX 0x40 requant)
out[co*P+p]= saturate_u8( max(r + bias[co], 0) )        (fused bias + ReLU + narrow)
```

This is the canonical on-device tiled inference conv — compose THREE mechanism groups:

- **im2col -> HMX matmul.** Lower the conv to a tiled matmul (M=P=256, K=C_in*Kh*Kw=144,
  N=C_out=64) in 32x32 crouton tiles. Pack the activation crouton via a division-free
  im2col (precompute `koff[k]` = input offset of reduction index k, `poff[p]` = base input
  offset of output position p); K=144 zero-pads to 5 crouton K-tiles. Weight value =
  `W[co*KK + kg]`. One `mxclracc` per output tile, then accumulate the K-tiles.
- **DMA + VTCM double-buffer.** Partition VTCM into disjoint regions: two double-buffered
  activation crouton slots (2KB x2), two weight slots (1KB x2), a requant-config tile, and
  an output crouton tile. For each K-tile, im2col-pack the crouton in cacheable DDR, then
  uDMA it DDR->VTCM; while HMX computes K-tile `kt` from slot `kt&1`, uDMA-prefetch K-tile
  `kt+1` into the alternate slot so DMA latency hides behind HMX compute. The Type-0 DMA
  descriptor must be static/global (a stack descriptor no-ops at -O2); chain activation ->
  weight via the descriptor next-ptr.
- **HVX fused epilogue.** Bulk-copy the HMX output crouton VTCM->cacheable (128B vector
  copies — scalar VTCM access is ~48 cyc), sign-extend the 12-bit requant field, add
  `bias[co]`, ReLU, saturate to uint8.

The harness has enabled the HMX context and installed an identity VTCM translation; use
VTCM scratch at `HVX_VTCM_BASE`. A competent HVX im2col+`vrmpy` conv with the same epilogue
is the baseline; the tiled HMX+DMA expert must beat it by >=1.2x kernel-cycles.


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
