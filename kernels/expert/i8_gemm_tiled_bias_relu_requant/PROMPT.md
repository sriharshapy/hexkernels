# int8 tiled GEMM + fused bias + ReLU + requant-to-uint8 (HMX + DMA + VTCM double-buffer)

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, const int32_t *bias, uint8_t *out, int M, int N, int K)`
computing a tiled row-major int8 GEMM on the HMX matrix engine, the HMX native
requant, and the classic quantized-inference epilogue (bias -> ReLU -> saturating
narrow to uint8), all fused. `M == N == K == 128`:

```
acc[i][j] = sum_k A[i*K+k] * B[k*N+j] (A uint8 0..3, B int8 -3..3)
r[i][j] = sign_extend_12bit((acc*17 + 8) >> 4) (0x40-config HMX requant field)
biased = r[i][j] + bias[j] (int32, per-output-column bias)
relu = biased > 0 ? biased : 0
out[i][j] = saturate_u8(relu) (clamp to [0,255])
```

This is the canonical on-device NPU inference kernel: it must COMPOSE three
mechanism groups.

- **HMX** for the matmul over a 4x4 grid of 32x32 output tiles, each accumulating
 over 4 K-tiles. Clear the accumulator once per output tile (`mxclracc`), issue
 the four K-tile `(activation, weight)` matmul packets, then the requant store.
- **DMA + VTCM double-buffer** to stream crouton tiles DDR->on-chip. Partition VTCM
 into DISJOINT regions: two double-buffered activation slots (2KB each), two weight
 slots (1KB each), a requant-config tile, and an output crouton tile. Pack the next
 K-tile's crouton in cacheable DDR, then start a Type-0 1D descriptor with `Q6_dmstart_A` to
 uDMA it into the ALTERNATE VTCM slot (async) while HMX computes the current tile;
 `Q6_R_dmwait()` before reusing the slot. The descriptor must be static/global
 (a stack descriptor silently no-ops at -O2). The harness has installed the identity
 VTCM `add_translation` and enabled the HMX context before calling you.
- **HVX** for the on-chip data movement (bulk 128B vector copies of the output
 crouton / config) and the fused epilogue.

Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte
of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B
tile, load limit 1023) — activation and weight need DIFFERENT load limits in one
packet; output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.

Input ranges keep `|r| < 2048` so the 12-bit requant field is exact (no HMX
saturation) and the output is bit-exact to the reference. A competent HVX `vrmpy`
tiled matmul + the same epilogue (no HMX, no DMA) is the baseline; the composed
expert must beat it by >=1.2x.


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
