# int8 SwiGLU gated FFN block — 3 matmuls + SiLU gate + product (HMX x3 + VTCM + HVX)

Implement
`candidate_kernel(const uint8_t *X, const int8_t *Wg, const int8_t *Wu, const int8_t *Wd, const int32_t *bd, const uint8_t *silu_lut, int8_t *out, int S, int D, int Dff)`
computing a SwiGLU gated feed-forward layer (LLaMA/PaLM-style): two parallel
projections, a SiLU-gated elementwise product, and a down projection, all int8.
`S == 64`, `D == 64`, `Dff == 128`. `X` is uint8 (0..3); `Wg`,`Wu` are int8
(-3..3); `Wd` is int8 ternary with zero-sum columns; `bd` is int32; `silu_lut` is
a 256-entry runtime SiLU (hardswish) LUT indexed by `g+128`:

```
gate_acc[i][j] = sum_{k<D} X[i*D+k] * Wg[k*Dff+j] (i<S, j<Dff)
up_acc[i][j] = sum_{k<D} X[i*D+k] * Wu[k*Dff+j]
g[i][j] = clamp( sx12((gate_acc*17+8)>>4) >> SW_SG , -128,127) (int8)
u[i][j] = clamp( sx12((up_acc *17+8)>>4) >> SW_SU , -128,127) (int8)
silu_g = silu_lut[g + 128] (SiLU, int8)
h[i][j] = clamp( (silu_g * u) >> SW_SH , -SW_ZP , SW_ZP-1 ) (SiLU-gated product)
Hu[i][j] = h[i][j] + SW_ZP (uint8 0..15)

out_acc[i][m] = sum_{j<Dff} Hu[i][j] * Wd[j*D+m] (i<S, m<D)
out[i][m] = sat_i8( ( sx12((out_acc*17+8)>>4) + bd[m] ) >> SW_SO ) (int8)
```

This is the canonical modern gated FFN: it must COMPOSE three mechanism groups and
get SwiGLU's structure right (SiLU on the GATE branch, then the elementwise
product with up, then the down projection).

- **HMX** for ALL THREE matmuls (gate, up, down), each over a grid of 32x32 output
 tiles accumulating over the K-tiles. Clear the accumulator once per output tile
 (`mxclracc`), issue the K-tile `(activation, weight)` matmul packets, then the
 `bias=mxmem` + `mxmem(...):after.uh=acc:2x1` requant store.
- **VTCM** for the gated intermediate activation H: keep H on-chip between the
 up/gate matmuls and the down matmul (bulk-copy it into VTCM after the gate,
 bulk-copy it back out to pack the down matmul's activation croutons) alongside
 the HMX crouton operand slots. The harness installed the identity VTCM
 `add_translation` and enabled the HMX context before calling you.
- **HVX** for the SiLU LUT gating + elementwise product + the requant/bias/saturate
 epilogues, and the bulk 128B vector copies of the crouton operands / output.

Crouton layouts (helpers in `harness_common.h`): activation = uint8 in the HIGH
byte of an fp16-crouton slot (2048B tile, load limit 2047); weight = 4-deep packed
int8 (1024B tile, load limit 1023) — activation and weight need DIFFERENT load
limits in one packet; output = crouton uint16, stored `mxmem(...):after.uh=acc:2x1`.
Stage crouton pack/unpack through cacheable buffers and move to/from VTCM in bulk
128B vector copies (a scalar VTCM access costs ~48 cyc in timing mode and loses).

The down-projection activation carries an int8 zero-point (`SW_ZP`) so the signed
gated value fits the unsigned HMX `activation.ub` load; because `Wd`'s columns sum
to zero the zero-point contributes no per-column offset, so `out_acc` equals the
true gated·Wd and every 12-bit requant field stays `< 2048` (exact). The SiLU LUT
makes the nonlinearity exact, so the whole block is BIT-EXACT to the scalar
reference. A competent HVX `vrmpy` SwiGLU (all three matmuls as dot-products, H in
plain DDR, no HMX) is the baseline; the composed expert must beat it by >=1.2x.


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
