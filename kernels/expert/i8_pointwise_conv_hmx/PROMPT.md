# int8 pointwise (1x1) convolution on the HMX matrix engine — deep-channel variant

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n)`
computing a 1x1 (pointwise) convolution with the HMX requant epilogue, for the fixed shape
`C_in = 96`, `C_out = 64`, `P = H*W = 64` spatial positions:

```
acc[co][p]  = sum_ci in[ci*P + p] * W[co*C_in + ci]   (in uint8 0..7, W int8 -2..2)
out[co*P+p] = ((acc*17 + 8) >> 4) & 0xFFF             (0x40-config requant, 12-bit field)
```

- Same lowering as `i8_conv2d_1x1_hmx` (pointwise conv over channels = a matmul) but with a
  **deeper channel reduction** (C_in=96 = 3 crouton K-tiles).
- The harness has already enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- Tile into 32x32 crouton output tiles; clear the accumulator once per output tile
  (`mxclracc`), then issue the 3 K-tile `(activation, weight)` matmul packets before the store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of an
  fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B, load limit
  1023, different limit than activation); output = crouton uint16, `mxmem(...):after.uh=acc:2x1`.
- Stage the crouton pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B
  vector copies (scalar VTCM access is expensive in timing mode).
- A competent HVX `vrmpy` pointwise conv is the baseline; the HMX expert must beat it by >=1.2x.


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
