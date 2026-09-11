# int8 grouped-query attention QK^T on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *Q, const int8_t *K, uint16_t *out)`
computing the GQA score matrices for `H_Q=2` query heads sharing `H_KV=1` key
head (`GROUP_SIZE = H_Q/H_KV = 2`), each head `S=64 x D=64`:

```
Q:   [H_Q  x S x D] uint8, row-major, head outermost.
K:   [H_KV x S x D] int8,  row-major (row j IS key vector j -- already K^T-shaped).
out: [H_Q  x S x S] uint16 (12-bit HMX requant field).

for h_q in 0..H_Q-1:
    h_kv = h_q / GROUP_SIZE
    scores[h_q][i][j] = sum_d Q[h_q,i,d] * K[h_kv,j,d]
    out[h_q][i][j]    = ((scores*17 + 8) >> 4) & 0xFFF     (0x40-config HMX requant)
```

- The harness has already enabled the HMX context before calling you; use VTCM
  scratch at `HVX_VTCM_BASE`.
- Output is the uint16 12-bit field, bit-exact to the scalar reference.
- **Exploit the GQA structure for real speedup**: pack the shared KV head's HMX
  weight croutons into VTCM **once**, up front, and leave them resident — the
  `GROUP_SIZE` query heads that share this KV head must NOT re-pack the weight;
  only the activation (Q) side is re-packed per head/tile.
- Per head, 64x64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates
  over two D-tiles (D=64 = 2*32). Clear the HMX accumulator once per output tile
  (`mxclracc`), then issue both D-tile `(activation, weight)` matmul packets before
  the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of
  an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B tile,
  load limit 1023) — activation and weight need DIFFERENT load limits in one packet.
- K's natural `[H_KV,S,D]` layout is already K^T relative to a generic matmul's B
  operand — pack weight tile `(tj,kt)` from `K[(tj*T+j)*D + (kt*T+k)]` (swapped
  tile indices), same trick as the single-head QK^T sibling task.
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
  through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- A competent HVX `vrmpy` matmul, run independently per query head against the
  shared KV head (no weight reuse — a plain per-head loop), is the baseline; the
  HMX expert's weight-crouton reuse must beat it by >=1.2x.


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
