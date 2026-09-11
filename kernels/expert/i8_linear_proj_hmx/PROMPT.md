# int8 linear projection (attention QKV/output projection) on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *X, const int8_t *W, const int32_t *bias, int8_t *out, int n)`
computing the row-major int8 linear projection, the HMX matrix engine's native
requant, and a fused per-output-channel bias epilogue (NO activation), for `n == 64`
(`S == Din == Dout == 64`):

```
acc[i][o] = sum_d X[i*n+d] * W[o*n+d] (X uint8 0..7, W int8 -3..3;
 W is [Dout,Din] nn.Linear
 row-major -- row o IS
 output-channel o's weight)
r[i][o] = sign_extend_12bit((acc*17 + 8) >> 4) (0x40-config HMX requant field)
biased = r[i][o] + bias[o] (int32, per-output-channel bias)
out[i][o] = saturate_i8(biased) (clamp to [-128,127])
```

- The harness has already enabled the HMX context before calling you; use VTCM
 scratch at `HVX_VTCM_BASE`.
- Output is int8, bit-exact to the scalar reference (saturating narrow). This is a
 plain projection — do NOT apply ReLU or any other activation.
- 64x64 is a 2x2 grid of 32x32 output tiles; each output tile accumulates over two
 Din-tiles (Din=64 = 2*32). Clear the HMX accumulator once per output tile
 (`mxclracc`), then issue both Din-tile `(activation, weight)` matmul packets
 before the requant store.
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH byte of
 an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed (1024B tile,
 load limit 1023) — activation and weight need DIFFERENT load limits in one packet.
- **W's natural `[Dout,Din]` nn.Linear layout is already W^T relative to a generic
 matmul's B operand** — pack the weight tile by reading `W[(tj*T+o)*Din + (kt*T+k)]`
 (swapped tile indices), not `W[(kt*T+k)*Din + (tj*T+o)]` (that computes X.W, not X.W^T).
- Scalar VTCM accesses are expensive in timing mode — stage the crouton pack/unpack
 through cacheable buffers and move to/from VTCM in bulk 128B vector copies.
- After unpacking the HMX output, sign-extend the 12-bit field, add `bias[o]`, then
 saturate to int8 — this chain is the fused HVX epilogue.
- A competent HVX `vrmpy` matmul (direct row-row dot against W — no transpose needed
 there either, since W's rows are already output channels) + the same epilogue is
 the baseline; the HMX expert must beat it by >=1.2x.


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
