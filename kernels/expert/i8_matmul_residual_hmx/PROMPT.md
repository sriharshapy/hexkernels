# int8 64x64 matrix multiply + fused full-matrix int32 residual add on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *C, int32_t *out, int n)`
computing the row-major int8 matmul, the HMX matrix-engine's native requant, and
an elementwise int32 residual add, all fused into the epilogue, for `n == 64`:

```
acc[i][j] = sum_k A[i*n+k] * B[k*n+j] (A uint8 0..7, B int8 -3..3)
r[i][j] = sign_extend_12bit((acc*17 + 8) >> 4) (0x40-config HMX requant field)
out[i][j] = r[i][j] + C[i*n+j] (int32, no saturation, no narrow)
```

- The harness has already enabled the HMX context before calling you; use VTCM
 scratch at `HVX_VTCM_BASE`. `C` is `[n*n]` int32, row-major, same shape as `out`
 -- a FULL matrix residual (not a per-output-column broadcast like the bias
 sibling task, and not a narrowing-saturate like the residual-requant sibling
 task: this task's output stays int32 with no clamp anywhere).
- Output is int32, bit-exact to the scalar reference.
- 64x64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over two
 K-tiles (K=64 = 2*32). Clear the HMX accumulator once per output tile (`mxclracc`),
 then issue both K-tile `(activation, weight)` matmul packets before the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH
 byte of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep
 packed (1024B tile, load limit 1023); output = crouton uint16, stored
 `mxmem(...):after.uh=acc:2x1`.
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
 through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- After unpacking the HMX output, sign-extend the 12-bit field, then add the
 matching `C` element (plain int32 add) — this is the fused HVX epilogue.
- A competent HVX `vrmpy` matmul + the same epilogue is the baseline; the HMX
 expert must beat it by >=1.2x.

`the HVX headers`. Do NOT write `main`. Respond with a single complete C
code block.


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
