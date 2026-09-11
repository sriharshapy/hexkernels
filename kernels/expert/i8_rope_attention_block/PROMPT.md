# int8 RoPE attention BLOCK (RoPE-rotate Q,K -> QK^T-HMX -> softmax-HVX -> A.V-HMX, VTCM-staged)

Implement
`candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V, const int8_t *cos_lut, const int8_t *sin_lut, uint16_t *out, int S, int D, int rot_shift, const uint8_t *exp_lut)`
computing a full single-head attention layer over `S=64` tokens of head-dim `D=64`
with a **Rotary Position Embedding (RoPE)** pre-rotation applied to `Q` and `K`.
`Q` is uint8 (0..7), `K` and `V` are int8 (-3..3), `HALF = D/2 = 32`:

```
RoPE (per row i, per feature pair (d, d+HALF)):
  c = cos_lut[i*HALF+d]; s = sin_lut[i*HALF+d]           (int8 runtime tables)
  Qr[i*D+d]      = clamp_u8( (Q[i*D+d]*c - Q[i*D+d+HALF]*s) >> rot_shift )
  Qr[i*D+d+HALF] = clamp_u8( (Q[i*D+d]*s + Q[i*D+d+HALF]*c) >> rot_shift )
  Kr[...]        = clamp_i8( same rotation on K )         (rot_shift = 4)

scores[i][j] = sum_d Qr[i*D+d] * Kr[j*D+d]                (rotated Q.K^T)
s12[i][j]    = ((scores*17 + 8) >> 4) & 0xFFF             (0x40-config HMX requant field)
scaled[i][j] = clamp( sign_extend_12bit(s12) >> 4 , -128, 127)   (attention SCALE)
probs[i][:]  = softmax_lut( scaled[i][:] ) over the KEY axis j    (uint8, row sum ~255)
                 m   = max_j scaled[i][j]
                 e_j = exp_lut[ clamp(scaled[i][j]-m, -255, 0) + 255 ]
                 probs[i][j] = (e_j*255 + (sum_j e_j)/2) / (sum_j e_j)
out_raw[i][d]= sum_j probs[i][j] * V[j*D+d]                       (probs uint8, V int8)
out[i*D+d]   = ((out_raw*17 + 8) >> 4) & 0xFFF                    (0x40-config HMX requant)
```

This is a marquee L3 kernel: it COMPOSES three mechanism groups.

- **HVX/scalar** applies the RoPE rotation to `Q` and `K` (the two halves of the head
  dimension) before the matmul, then runs the scale + row-wise softmax between the
  matmuls, plus the bulk 128B vector copies that stage crouton tiles to/from VTCM.
- **HMX** runs both matmuls (rotated `QK^T` and `A.V`) on the matrix engine, each a 2x2
  grid of 32x32 crouton output tiles accumulating over 2 reduction tiles. `mxclracc`
  once per output tile; `QK^T` packs the weight from `Kr` stored `[S,D]` (key row j is
  key vector j) with swapped tile indices; `A.V` packs `V` as a generic `B[K,N]`
  operand (no transpose), reducing over the key axis.
- **VTCM** holds the HMX operand/output crouton tiles (`HVX_VTCM_BASE`). Crouton
  pack/unpack is staged in cacheable DDR and moved to/from VTCM in bulk vector copies
  (a scalar VTCM access costs ~48 cyc in timing mode). The harness enables the HMX
  context and installs the identity VTCM translation before calling you.

Crouton layouts (helpers in `harness_common.h`): activation = uint8 in the HIGH byte of
an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed int8 (1024B
tile, load limit 1023) — activation and weight need DIFFERENT load limits in one packet;
output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.

The whole block is fixed-point integer, so it is BIT-EXACT to the scalar reference (no
tolerance). Input ranges + `rot_shift=4` keep every requant field `< 2048`. A competent
HVX implementation of the whole block (RoPE + both matmuls + softmax, no HMX) is the
baseline; the composed expert must beat it by >=1.2x (measured 1.58x). Do NOT hardcode
the cos/sin/exp tables — they are runtime inputs.


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
