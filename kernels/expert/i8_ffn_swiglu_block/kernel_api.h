#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/* L3 int8 SwiGLU FFN BLOCK -- the gated feed-forward layer of a modern
 * transformer (LLaMA/PaLM-style). THREE matmuls with a SiLU-gated elementwise
 * product between them. Composes THREE mechanism groups:
 *   - HMX matrix engine for ALL THREE matmuls (gate, up, down; 32x32 croutons),
 *   - VTCM for the HMX crouton operand slots + the VTCM-resident gated
 *     intermediate activation H between the up/gate matmuls and the down matmul,
 *   - HVX for the SiLU LUT gating, elementwise product, requant epilogues, and
 *     the bulk on-chip data movement.
 *
 * Given X [S x D] (uint8 0..3), gate/up weights Wg,Wu [D x Dff] (int8 -3..3),
 * down weight Wd [Dff x D] (int8 ternary, columns zero-sum), bias bd [D] (int32),
 * and a runtime SiLU LUT silu_lut[256] (hardswish approximation, index g+128):
 *
 *   gate_acc[i][j] = sum_{k<D} X[i*D+k] * Wg[k*Dff+j]     (i<S, j<Dff)
 *   up_acc[i][j]   = sum_{k<D} X[i*D+k] * Wu[k*Dff+j]
 *   g[i][j]        = clamp( sx12((gate_acc*17+8)>>4) >> SW_SG , -128, 127)   (int8)
 *   u[i][j]        = clamp( sx12((up_acc  *17+8)>>4) >> SW_SU , -128, 127)   (int8)
 *   silu_g         = silu_lut[g + 128]                                       (SiLU, int8)
 *   h[i][j]        = clamp( (silu_g * u) >> SW_SH , -SW_ZP , SW_ZP-1 )       (gated, signed)
 *   Hu[i][j]       = h[i][j] + SW_ZP                                          (uint8 0..2*ZP-1)
 *
 *   out_acc[i][m]  = sum_{j<Dff} Hu[i][j] * Wd[j*D+m]     (i<S, m<D)
 *   out[i][m]      = sat_i8( ( sx12((out_acc*17+8)>>4) + bd[m] ) >> SW_SO )   (int8)
 *
 * SwiGLU's defining ops: SiLU applied to the GATE projection (not up), then the
 * elementwise PRODUCT SiLU(gate)*up, then the down projection. The down-proj
 * activation carries an int8 zero-point SW_ZP (standard int8 quant); because Wd's
 * columns sum to zero the zero-point contributes NO per-column offset, so
 * out_acc equals the true gated.Wd and every 12-bit HMX requant field is exact
 * -> the whole block is BIT-EXACT to the scalar reference. S=64, D=64, Dff=128
 * (all 32-multiples). The harness enables the HMX context AND installs an identity
 * VTCM translation before the timed call. */
#define SW_S   64
#define SW_D   64
#define SW_DFF 128
#define SW_SG  2   /* gate scale right-shift  */
#define SW_SU  2   /* up   scale right-shift  */
#define SW_SH  10  /* SiLU-gated product right-shift */
#define SW_ZP  8   /* down-proj activation int8 zero-point (h in [-8,7]) */
#define SW_SO  3   /* output requant right-shift */

void candidate_kernel(const uint8_t *X, const int8_t *Wg, const int8_t *Wu,
                      const int8_t *Wd, const int32_t *bd, const uint8_t *silu_lut,
                      int8_t *out, int S, int D, int Dff);

/* Shared fixed-point pipeline (identical across ref/baseline/expert). */
static inline int sw_sx12(int field) {   /* sign-extend HMX 12-bit requant field */
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline int sw_clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}
/* scale a 12-bit requant field down to int8 (gate / up projection outputs). */
static inline int sw_scale(int field, int sh) { return sw_clamp_i8(sw_sx12(field) >> sh); }
/* SiLU-gated product -> signed gated activation h in [-SW_ZP, SW_ZP-1]. */
static inline int sw_gate(int silu_g, int u) {
    int h = (silu_g * u) >> SW_SH;
    if (h >  SW_ZP - 1) h =  SW_ZP - 1;
    if (h < -SW_ZP)     h = -SW_ZP;
    return h;
}
/* zero-pointed unsigned down-proj activation. */
static inline uint8_t sw_hu(int silu_g, int u) { return (uint8_t)(sw_gate(silu_g, u) + SW_ZP); }
static inline signed char sw_sat_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (signed char)v;
}
#endif
