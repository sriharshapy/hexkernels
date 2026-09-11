# int8 1x1 (pointwise) convolution on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *in, const int8_t *W, uint16_t *out, int n)`
computing a 1x1 (pointwise) convolution with the HMX requant epilogue, for the fixed shape
`C_in = C_out = 64`, `P = H*W = 64` spatial positions:

```
acc[co][p] = sum_ci in[ci*P + p] * W[co*C_in + ci] (in uint8 0..7, W int8 -3..3)
out[co*P + p] = ((acc*17 + 8) >> 4) & 0xFFF (0x40-config requant, 12-bit field)
```

- A 1x1 conv over channels **is a matmul**: rows = spatial positions (M=P), reduction =
 input channels (K=C_in), cols = output channels (N=C_out). No im2col gather is needed for
 1x1 (each output position reads one input pixel per channel); the only conv-specific work
 is the NCHW<->position-major gather during the crouton pack.
- The harness has already enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- Output is the uint16 12-bit two's-complement requant field, bit-exact to the scalar reference.
- Tile into 32x32 crouton output tiles; clear the HMX accumulator once per output tile
 (`mxclracc`), then issue the C_in/32 K-tile `(activation, weight)` matmul packets before
 the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of an
 fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B tile, load
 limit 1023) — activation and weight need DIFFERENT load limits in one packet; output =
 crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack through
 cacheable buffers and move to/from VTCM in bulk 128B vector copies.
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
