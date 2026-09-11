# int8 64x64 matrix multiply + fused bias-add + GELU-via-LUT on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias, const int8_t *gelu_lut, int8_t *out, int n)`
computing the row-major int8 matmul, the HMX matrix-engine's native requant, a
per-column bias add, saturation, and a GELU LUT lookup, all fused into the
epilogue, for `n == 64`:

```
acc[i][j] = sum_k A[i*n+k] * B[k*n+j]               (A uint8 0..7, B int8 -3..3)
r[i][j]   = sign_extend_12bit((acc*17 + 8) >> 4)     (0x40-config HMX requant field)
biased    = r[i][j] + bias[j]                         (int32, per-output-column bias)
pre_lut   = saturate_i8(biased)                        (clamp to [-128,127])
out[i][j] = gelu_lut[(uint8_t)(pre_lut + 128)]          (256-entry int8 LUT)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- `gelu_lut` is a RUNTIME parameter (256 int8 entries, index = `pre_lut + 128`) —
  do not hardcode it; the output is bit-exact to the scalar reference because the
  LUT itself is the exact contract (no float tolerance involved).
- 64x64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over two
  K-tiles (K=64 = 2*32). Clear the HMX accumulator once per output tile (`mxclracc`),
  then issue both K-tile `(activation, weight)` matmul packets before the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of
  an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B tile,
  load limit 1023) — activation and weight need DIFFERENT load limits in one packet;
  output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
  through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- After unpacking the HMX output, sign-extend the 12-bit field, add `bias[j]`,
  saturate to int8, then look up the LUT — this chain is the fused HVX epilogue.
- A competent HVX `vrmpy` matmul + the same epilogue is the baseline; the HMX expert
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
