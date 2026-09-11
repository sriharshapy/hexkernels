# int8 32x32 HMX matmul + requantize-to-int8 (saturating)

Implement `candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n)`
for `n == 32`:

```
acc[i][j] = sum_{k=0}^{n-1} A[i*n+k] * B[k*n+j]      (A uint8 0..15, B int8 -4..4)
r[i][j]   = sign_extend_12bit( (acc*17 + 8) >> 4 )    (0x40-config HMX requant field)
out[i][j] = clamp(r[i][j], -128, 127)                  (signed int8, SATURATING)
```

- The harness has enabled the HMX context; use VTCM scratch at `HVX_VTCM_BASE`.
- Standard 32x32 int8 HMX tile: activation crouton (int8 in HIGH byte, 2048B,
  lim 2047), 4-deep weight tile (1024B, lim 1023); `mxclracc`; one matmul packet;
  `bias=mxmem` (0x40 config); store `:after.uh=acc:2x1`; unpack + sign-extend the
  12-bit field.
- Narrow to int8 WITH saturation: the wider input range makes `|r|` exceed 127
  for many outputs, so a plain truncating cast is wrong -- clamp to [-128, 127].
- Stage crouton pack/unpack through cacheable buffers + bulk 128B vector copies;
  beat a competent HVX `vrmpy`+requant baseline by >=1.2x.
- Layout helpers: `hvx_hmx_i8_act_off`, `hvx_hmx_i8_wgt_off`, `hvx_hmx_i8_out_off`.

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
