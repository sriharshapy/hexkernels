/* EXPERT (achievability bar) / Solution 1 -- HVX word-lane requantize.
 * Per 32-bit lane: v = acc*mult; round-half-away-from-zero by shift; saturate
 * to int8. |acc|<=10,000,000 and |mult|<=200 keep |acc|*|mult| < 2^31, so
 * decomposing into sign * |acc|*|mult| and doing the magnitude multiply via
 * Q6_Vw_vmpyie_VwVuh (word x unsigned-halfword) is exact (same proven pattern
 * as bias_add_requant/requantize_i32_i8_dma's requant_word helper, here without
 * the bias-sum or zero-point). Four int32 result vectors (128 elements) are
 * narrowed to one int8 vector via saturating packs, which also perform the
 * final clamp to [-128,127]. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector requant_word(HVX_Vector w, HVX_Vector vamult, int mult_neg,
                                      int shift, HVX_Vector vhalf) {
    HVX_Vector sm = Q6_Vw_vasr_VwR(w, 31);                          /* sign(w): 0 or -1 */
    HVX_Vector aw = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(w, sm), sm);       /* |w| */
    HVX_Vector am = Q6_Vw_vmpyie_VwVuh(aw, vamult);                 /* |w|*|mult| */
    HVX_Vector sh = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    HVX_Vector sv = mult_neg ? Q6_V_vnot_V(sm) : sm;                /* sign(v) = sign(w)^sign(mult) */
    return Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sv), sv);
}

void candidate_kernel(const int32_t *acc, int8_t *out, int n, int32_t mult, int shift) {
    int half  = shift > 0 ? (1 << (shift - 1)) : 0;
    int amult = mult < 0 ? -mult : mult;
    int mneg  = mult < 0;
    HVX_Vector vamult = Q6_V_vsplat_R((uint32_t)(uint16_t)amult);
    HVX_Vector vhalf  = Q6_V_vsplat_R((uint32_t)half);
    const int W = 32; /* int32 lanes per vector */
    int i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        HVX_Vector r0 = requant_word(*(const HVX_Vector*)(acc+i),      vamult, mneg, shift, vhalf);
        HVX_Vector r1 = requant_word(*(const HVX_Vector*)(acc+i+W),    vamult, mneg, shift, vhalf);
        HVX_Vector r2 = requant_word(*(const HVX_Vector*)(acc+i+2*W),  vamult, mneg, shift, vhalf);
        HVX_Vector r3 = requant_word(*(const HVX_Vector*)(acc+i+3*W),  vamult, mneg, shift, vhalf);
        HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(r1, r0);  /* i16: [0..31]=r0, [32..63]=r1 */
        HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
        HVX_Vector b   = Q6_Vb_vpack_VhVh_sat(h23, h01);/* i8:  [0..63]=h01, [64..127]=h23 */
        *(HVX_Vector*)(out + i) = b;
    }
    for (; i < n; i++) {
        int64_t v  = (int64_t)acc[i] * (int64_t)mult;
        int64_t hf = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r  = (v >= 0) ? ((v + hf) >> shift) : -(((-v) + hf) >> shift);
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
