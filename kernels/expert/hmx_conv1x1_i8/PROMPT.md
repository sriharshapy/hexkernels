# int8 1x1 convolution (channel matmul) + per-channel bias on the HMX matrix engine

Implement `candidate_kernel(const uint8_t *In, const int8_t *W, const int32_t
*bias, int32_t *out, int n)` for `n == 32`. A 1x1 convolution contracts the
channel axis:

```
acc[p][co]  = sum_ci In[p*n+ci] * W[co*n+ci]
out[p*n+co] = sign_extend_12bit((acc*17 + 8) >> 4) + bias[co]
```

`In` is `[P x Cin]` activations (uint8 0..7), `W` is `[Cout x Cin]` weights (int8
-3..3), `bias` is `[Cout]` int32; `out = In*W^T + bias`. The `(acc*17+8)>>4 &
0xFFF` requant is the HMX matrix engine's native output at the `0x40` bias-config.

- The harness has already enabled the HMX context; use VTCM scratch at
  `HVX_VTCM_BASE`. int8 activation in the HIGH byte of an fp16-crouton slot
  (`hvx_hmx_i8_act_off`), weight 4-deep packed and read from W TRANSPOSED
  (`aWgt[hvx_hmx_i8_wgt_off(ci,co)] = W[co][ci]`), output crouton read as uint16
  (`hvx_hmx_i8_out_off`).
- Sequence: `mxclracc` -> `{ activation.ub=mxmem(...); weight.b=mxmem(...) }` ->
  `bias=mxmem(0x40-fill)` -> `mxmem(...):after.uh=acc:2x1` -> isync. Sign-extend
  the 12-bit field and add `bias[co]` in the un-crouton epilogue.
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
