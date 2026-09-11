/* EXPERT (achievability bar) -- same as solutions/s2.c: 4-vectors-per-
 * iteration unrolling of the sign/magnitude requantize primitive (128
 * elements -> 2 requantized int16 output vectors per iteration), with a
 * 64-element (2-vector) cleanup block and a scalar tail. */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector requant_word(HVX_Vector w, HVX_Vector vmult, int shift, HVX_Vector vhalf) {
    HVX_Vector sm = Q6_Vw_vasr_VwR(w, 31);
    HVX_Vector aw = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(w, sm), sm);
    HVX_Vector am = Q6_Vw_vmpyie_VwVuh(aw, vmult);
    HVX_Vector sh = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    return Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sm), sm);
}

void candidate_kernel(const int32_t *raw, int16_t *out, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int total = M * N;
    int half  = (scale_shift > 0) ? (1 << (scale_shift - 1)) : 0;
    HVX_Vector vmult = Q6_V_vsplat_R((uint32_t)(uint16_t)scale_mult);
    HVX_Vector vhalf = Q6_V_vsplat_R((uint32_t)half);
    const int W = 32; /* int32 lanes per vector */

    int i = 0;
    for (; i + 4*W <= total; i += 4*W) {
        HVX_Vector r0 = requant_word(*(const HVX_Vector *)(raw + i),       vmult, scale_shift, vhalf);
        HVX_Vector r1 = requant_word(*(const HVX_Vector *)(raw + i + W),   vmult, scale_shift, vhalf);
        HVX_Vector r2 = requant_word(*(const HVX_Vector *)(raw + i + 2*W), vmult, scale_shift, vhalf);
        HVX_Vector r3 = requant_word(*(const HVX_Vector *)(raw + i + 3*W), vmult, scale_shift, vhalf);
        HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(r1, r0);  /* i16: [0..31]=r0, [32..63]=r1 */
        HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(r3, r2);  /* i16: [0..31]=r2, [32..63]=r3 */
        *(HVX_Vector *)(out + i)      = h01;
        *(HVX_Vector *)(out + i + 64) = h23;
    }
    /* remaining full 64-element (2-vector) block, if any. */
    for (; i + 2*W <= total; i += 2*W) {
        HVX_Vector r0 = requant_word(*(const HVX_Vector *)(raw + i),     vmult, scale_shift, vhalf);
        HVX_Vector r1 = requant_word(*(const HVX_Vector *)(raw + i + W), vmult, scale_shift, vhalf);
        *(HVX_Vector *)(out + i) = Q6_Vh_vpack_VwVw_sat(r1, r0);
    }
    for (; i < total; i++) {
        int64_t r    = (int64_t)raw[i] * (int64_t)scale_mult;
        int64_t hf   = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
        int64_t q    = (r >= 0) ? ((r + hf) >> scale_shift) : -((-r + hf) >> scale_shift);
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
