# int8 32x32 matrix multiply on the HMX matrix engine, requantized to int8

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n)`
computing the row-major int8 matrix product with the HMX requant epilogue, narrowed
to a real int8 output, for `n == 32`:

```
acc[i][j] = sum_k A[i*n+k] * B[k*n+j]         (A uint8 0..3, B int8 -1..1)
r         = (acc*17 + 8) >> 4                 (0x40-config requant, 12-bit field)
out[i*n+j] = (int8_t) r                       (input ranges guarantee |r| <= 127)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- Unlike other HMX matmul tasks that expose the raw uint16 12-bit field, this
  task's output is the sign-narrowed int8 value itself, bit-exact to the scalar
  reference. The 12-bit field is a two's-complement value zero-extended into a
  uint16 -- sign-extend from bit 11 before narrowing to int8.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH
  byte of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep
  packed (1024B tile, load limit 1023) -- activation and weight need DIFFERENT
  load limits in one packet; output = crouton uint16, stored
  `mxmem(...):after.uh=acc:2x1`.
- Scalar VTCM accesses are expensive in timing mode -- stage the crouton
  pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B
  vector copies.
- A competent HVX `vrmpy` matmul (with the same requant-to-int8 epilogue) is the
  baseline; the HMX expert must beat it by >=1.2x.


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
