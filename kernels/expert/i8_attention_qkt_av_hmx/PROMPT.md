# Fused int8 attention (QK^T then AV, no softmax) on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V, int32_t *out, int S, int D)`
for `S == 32`, `D == 32` (Q, K, V are all `[S][D]` row-major):

```
scores[i][j] = sum_d Q[i*D+d] * K[j*D+d] (Q uint8 in {0,1}, K int8 in {0,1})
Sc[i][j] = sign_extend_12bit((scores*17 + 8) >> 4) (0x40-config HMX requant field)
acc[i][d] = sum_j Sc[i][j] * V[j*D+d] (V int8 in {-1,0,1})
out[i][d] = sign_extend_12bit((acc*17 + 8) >> 4) (0x40-config HMX requant field)
```

**No softmax.** This is deliberately a "linear attention" variant: `Sc` (the
QK^T requant output) is re-fed directly as the activation of the SECOND HMX
matmul against V, without a softmax normalization step in between. Softmax has
no exact fixed-point/HMX formulation that stays bit-exact, so it is out of
scope for this task; the point of the exercise is chaining two dependent HMX
matmuls through an intermediate int8 restage, not the softmax nonlinearity.

- Input ranges are pinned narrow so BOTH matmul stages stay in the exact
 (non-saturating) region of the 12-bit requant field, AND so `Sc` (which must
 be re-packed as an unsigned int8 HMX activation) is always non-negative:
 `Q,K in {0,1}` keeps `scores` in `[0,32]` so `Sc` is a small non-negative
 value in `[0,34]`; `V in {-1,0,1}` then keeps `|acc| <= 32*34*1 = 1088`, so
 `|out| <= 1157 < 2048` (field stays exact).
- Stage 1 (QK^T): `K` is stored `[S,D]` row-major (row j IS key vector j), so
 pack the weight tile with SWAPPED indices (`weight[k][j] = K[j][k]`) -- no
 separate transpose buffer needed, same trick as the QK^T-only sibling task.
 One `mxclracc`, one matmul packet (S=D=32 is a single crouton tile), one
 requant store, then unpack + sign-extend + cast to `uint8_t` -- this is `Sc`.
- Stage 2 (AV): re-pack `Sc` as the activation (int8 HIGH byte of a crouton
 slot) and `V` as the weight -- `V` is already in the generic `[K,N]` matmul
 layout (row j is value vector j), so NO index swap is needed here (unlike
 stage 1). One `mxclracc`, one matmul packet, one requant store, unpack +
 sign-extend -> `out` (int32).
- The harness has already enabled the HMX context before calling you; use VTCM
 scratch at `HVX_VTCM_BASE` for BOTH stages (reuse the same VTCM addresses
 sequentially -- stage 1 fully completes and is unpacked into a local buffer
 before stage 2 starts packing).
- Crouton layouts (helpers in `harness_common.h`): activation = int8 in the HIGH
 byte of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep
 packed (1024B tile, load limit 1023); output = crouton uint16, stored
 `mxmem(...):after.uh=acc:2x1`.
- Scalar VTCM accesses are expensive in timing mode — stage all crouton
 pack/unpack through cacheable buffers and move to/from VTCM in bulk 128B
 vector copies.
- Unpack the second stage's crouton output and sign-extend its 12-bit requant field
 **in HVX**, not in a scalar loop — that unpack is the fused HVX epilogue. One 128B
 vector is exactly one crouton row PAIR (`off(r,c) = (r/2)*64 + c*2 + (r&1)`), so
 `vshuffe`/`vshuffo` separate the two rows and a left-then-right shift pair sign-extends
 the field before widening to int32.
- A competent HVX `vrmpy` two-stage matmul (same requant chain, no softmax) is
 the baseline; the HMX expert must beat it by >=1.2x.

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
