# int8 attention QK^T scores on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out, int S, int D)`
computing the attention QK^T score matrix and the HMX matrix engine's native
requant, for `S == D == 64`:

```
scores[i][j] = sum_d Q[i*D+d] * K[j*D+d]      (Q uint8 0..7, K int8 -3..3;
                                                K is stored [S,D] row-major --
                                                row j IS key vector j, so this
                                                is Q . K^T with no explicit
                                                transpose of K)
out[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF  (0x40-config HMX requant field)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- Output is the uint16 12-bit field, bit-exact to the scalar reference.
- 64x64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over two
  D-tiles (D=64 = 2*32). Clear the HMX accumulator once per output tile (`mxclracc`),
  then issue both D-tile `(activation, weight)` matmul packets before the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of
  an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B tile,
  load limit 1023) — activation and weight need DIFFERENT load limits in one packet.
- **K's natural [S,D] layout is already K^T relative to a generic A.B matmul's B
  operand** — pack the weight tile by reading `K[(tj*T+j)*D + (kt*T+k)]` (swapped
  tile indices), not `K[(kt*T+k)*D + (tj*T+j)]` (that computes Q.K, not Q.K^T).
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
  through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- A competent HVX `vrmpy` dot-product matmul (direct row-row dot, same K-layout
  trick — no transpose needed there either) is the baseline; the HMX expert must
  beat it by >=1.2x.


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
