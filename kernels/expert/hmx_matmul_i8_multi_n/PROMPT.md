# int8 32x64 matmul over multiple N-tiles on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int32_t *out,
int m, int ncol, int k)` for `m == 32`, `k == 32`, `ncol == 64`:

```
acc[i][j] = sum_kk A[i*k+kk] * B[kk*ncol+j]
out[i][j] = sign_extend_12bit((acc*17 + 8) >> 4)      (int32, i<32, j<64)
```

`A` is uint8 (0..7), `B` is int8 (-3..3). The output width `ncol == 64` spans TWO
32-wide crouton output tiles, so run the 32x32 HMX matmul once per N-tile: pack
the shared activation once, then for each N-tile `tj` pack weight columns
`tj*32 .. tj*32+31`, matmul (mxclracc per tile), `0x40`-requant store, and
un-crouton into output columns `tj*32 ..`.

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE`. int8 activation in the HIGH byte of a crouton slot
  (`hvx_hmx_i8_act_off`), weight 4-deep packed (`hvx_hmx_i8_wgt_off`), output
  crouton read as uint16 (`hvx_hmx_i8_out_off`).
- Per tile: `mxclracc` -> `{ activation.ub=mxmem(...); weight.b=mxmem(...) }` ->
  `bias=mxmem(0x40-fill)` -> `mxmem(...):after.uh=acc:2x1` -> isync, then
  sign-extend the 12-bit field.
- Stage crouton pack/unpack through cacheable buffers + bulk 128B copies; the HMX
  expert must beat the HVX vrmpy baseline by >=1.2x.

Implement ONLY this function (Hexagon HVX C). Include `<hexagon_types.h>` and
`<hexagon_protos.h>`. Do NOT write `main()`. Respond with a single complete C
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
