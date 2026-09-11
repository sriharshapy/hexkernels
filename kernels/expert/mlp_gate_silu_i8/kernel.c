/* EXPERT (fastest measured solution -- see solutions/s1.c, identical). *
 * HVX SwiGLU gating step (elementwise), N=150.
 *
 * Fully-fused vectorized body per 128-lane block:
 *   1. idx  = gate XOR 0x80 (byte-wise; equals (uint8_t)(gate+128) -- adding
 *      128 mod 256 only flips the top bit since the low 7 bits of the addend
 *      are 0)
 *   2. g    = silu_lut[idx] via the standard vlut32/vlut32or 256-entry byte
 *      LUT gather (8 passes; matches datasets/v6/tasks/hvx_vlut8/expert.c
 *      and i8_lut_sigmoid/expert.c).
 *   3. widen g, up to int16 SEQUENTIALLY via Q6_Wh_vunpack_Vb (Q6_Wh_vmpy_VbVb
 *      DEINTERLEAVES and would be wrong here -- see i8_depthwise_conv2d_requant
 *      /expert.c's documented investigation); elementwise int16 multiply
 *      (safe: |127*127|=16129 << 32767); widen to int32 via Q6_Ww_vunpack_Vh.
 *   4. per-int32-lane round-half-away-from-zero requantize via the proven
 *      sign/magnitude decomposition (Q6_Vw_vmpyie_VwVuh) from
 *      epilogue_requant_i32_i8/expert.c's requant_word helper.
 *   5. two-step saturating pack (int32->int16->int8) performs the final
 *      clamp to [-128,127].
 * The 22-element tail (150 = 1*128 + 22) is handled by a plain scalar loop
 * using the same pinned formula -- see solutions/s2.c for a structurally
 * different (masked-vector-store) tail technique.
 */
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <stdint.h>

static inline HVX_Vector lut8_body(HVX_Vector vidx, HVX_Vector sTab0, HVX_Vector sTab1) {
    HVX_Vector res;
    res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
    res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);
    return res;
}

static inline HVX_Vector requant_word(HVX_Vector w, HVX_Vector vamult, int mult_neg,
                                      int shift, HVX_Vector vhalf) {
    HVX_Vector sm = Q6_Vw_vasr_VwR(w, 31);                          /* sign(w): 0 or -1 */
    HVX_Vector aw = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(w, sm), sm);       /* |w| */
    HVX_Vector am = Q6_Vw_vmpyie_VwVuh(aw, vamult);                 /* |w|*|mult| */
    HVX_Vector sh = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    HVX_Vector sv = mult_neg ? Q6_V_vnot_V(sm) : sm;                /* sign(v) = sign(w)^sign(mult) */
    return Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sv), sv);
}

void candidate_kernel(const int8_t *gate, const int8_t *up, const int8_t *silu_lut,
                      int8_t *out, int N, int32_t scale_mult, int scale_shift)
{
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(silu_lut));
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(silu_lut + 128));
    HVX_Vector v80   = Q6_V_vsplat_R(0x80808080);

    int half  = scale_shift > 0 ? (1 << (scale_shift - 1)) : 0;
    int amult = scale_mult < 0 ? -scale_mult : scale_mult;
    int mneg  = scale_mult < 0;
    HVX_Vector vamult = Q6_V_vsplat_R((uint32_t)(uint16_t)amult);
    HVX_Vector vhalf  = Q6_V_vsplat_R((uint32_t)half);

    int i = 0;
    for (; i + 128 <= N; i += 128) {
        HVX_Vector vgate = *(const HVX_Vector *)(gate + i);
        HVX_Vector vup   = *(const HVX_Vector *)(up + i);
        HVX_Vector vidx  = Q6_V_vxor_VV(vgate, v80);
        HVX_Vector vg    = lut8_body(vidx, sTab0, sTab1);

        HVX_VectorPair gw = Q6_Wh_vunpack_Vb(vg);
        HVX_VectorPair uw = Q6_Wh_vunpack_Vb(vup);
        HVX_Vector glo = Q6_V_lo_W(gw), ghi = Q6_V_hi_W(gw);
        HVX_Vector ulo = Q6_V_lo_W(uw), uhi = Q6_V_hi_W(uw);

        HVX_Vector plo = Q6_Vh_vmpyi_VhVh(glo, ulo);
        HVX_Vector phi = Q6_Vh_vmpyi_VhVh(ghi, uhi);

        HVX_VectorPair wplo = Q6_Ww_vunpack_Vh(plo);
        HVX_VectorPair wphi = Q6_Ww_vunpack_Vh(phi);
        HVX_Vector w0 = Q6_V_lo_W(wplo), w1 = Q6_V_hi_W(wplo);
        HVX_Vector w2 = Q6_V_lo_W(wphi), w3 = Q6_V_hi_W(wphi);

        HVX_Vector r0 = requant_word(w0, vamult, mneg, scale_shift, vhalf);
        HVX_Vector r1 = requant_word(w1, vamult, mneg, scale_shift, vhalf);
        HVX_Vector r2 = requant_word(w2, vamult, mneg, scale_shift, vhalf);
        HVX_Vector r3 = requant_word(w3, vamult, mneg, scale_shift, vhalf);

        HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(r1, r0);
        HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
        HVX_Vector b   = Q6_Vb_vpack_VhVh_sat(h23, h01);
        *(HVX_Vector *)(out + i) = b;
    }
    for (; i < N; i++) {
        uint8_t idx = (uint8_t)((int)gate[i] + 128);
        int32_t g = silu_lut[idx];
        int32_t prod = g * (int32_t)up[i];
        int64_t r  = (int64_t)prod * (int64_t)scale_mult;
        int64_t hf = (scale_shift > 0) ? ((int64_t)1 << (scale_shift - 1)) : 0;
        int64_t q  = (r >= 0) ? ((r + hf) >> scale_shift) : -(((-r) + hf) >> scale_shift);
        if (q >  127) q =  127;
        if (q < -128) q = -128;
        out[i] = (int8_t)q;
    }
}
