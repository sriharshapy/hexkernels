/* EXPERT (achievability bar) — HVX int32->int8 bias-add + requant.
 * Per 32-bit lane: sum = a+bias; v = sum*mult; round-half-away-from-zero by
 * shift; +zp; saturate to int8. |sum| (<=~300) and |mult| (<=127) keep the
 * product < 2^31, so int32 lanes suffice (the reference's int64 is defensive).
 *
 * Uniform scalar x word multiply uses Q6_Vw_vmpyie_VwVuh(v, splat(|mult|)):
 * Vd.w[i] = Vu.w[i] * Vv.w[i].uh[0], i.e. each lane times the UNSIGNED low
 * halfword of the splat -> handles |mult| only, so mult's sign is folded into
 * the sign-aware rounding (sign(v) = sign(sum) XOR sign(mult)). (Note:
 * Q6_Vw_vmpyi_VwRb rotates Rt's 4 bytes across lanes -> wrong for a scalar.)
 * Four int32 result vectors (128 elements) are narrowed to one int8 vector via
 * saturating packs (SEQUENTIAL Q6_V*_vpack_*_sat), which reproduce the clamp. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector requant_vec(HVX_Vector va, HVX_Vector vb,
                                     HVX_Vector vamult, int mult_neg, int shift,
                                     HVX_Vector vhalf, HVX_Vector vzp) {
    HVX_Vector vsum = Q6_Vw_vadd_VwVw(va, vb);
    HVX_Vector sm   = Q6_Vw_vasr_VwR(vsum, 31);                       /* sign(sum): 0 or -1 */
    HVX_Vector asum = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(vsum, sm), sm);    /* |sum| */
    HVX_Vector am   = Q6_Vw_vmpyie_VwVuh(asum, vamult);              /* |sum|*|mult| (low32) */
    HVX_Vector sh   = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    HVX_Vector sv   = mult_neg ? Q6_V_vnot_V(sm) : sm;               /* sign(v) mask */
    HVX_Vector r    = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sv), sv);     /* apply sign */
    return Q6_Vw_vadd_VwVw(r, vzp);                                  /* +zp (clamp done by pack) */
}

void candidate_kernel(const int32_t *a, const int32_t *bias, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    int half   = shift > 0 ? (1 << (shift - 1)) : 0;
    int amult  = mult < 0 ? -mult : mult;
    int mneg   = mult < 0;
    HVX_Vector vamult = Q6_V_vsplat_R((uint32_t)(uint16_t)amult);
    HVX_Vector vhalf  = Q6_V_vsplat_R((uint32_t)half);
    HVX_Vector vzp    = Q6_V_vsplat_R((uint32_t)(int32_t)zp);
    const int W = 32; /* int32 lanes per vector */
    int i = 0;
    for (; i + 4*W <= n; i += 4*W) {
        HVX_Vector r0 = requant_vec(*(const HVX_Vector*)(a+i),      *(const HVX_Vector*)(bias+i),      vamult, mneg, shift, vhalf, vzp);
        HVX_Vector r1 = requant_vec(*(const HVX_Vector*)(a+i+W),    *(const HVX_Vector*)(bias+i+W),    vamult, mneg, shift, vhalf, vzp);
        HVX_Vector r2 = requant_vec(*(const HVX_Vector*)(a+i+2*W),  *(const HVX_Vector*)(bias+i+2*W),  vamult, mneg, shift, vhalf, vzp);
        HVX_Vector r3 = requant_vec(*(const HVX_Vector*)(a+i+3*W),  *(const HVX_Vector*)(bias+i+3*W),  vamult, mneg, shift, vhalf, vzp);
        HVX_Vector h01 = Q6_Vh_vpack_VwVw_sat(r1, r0);  /* i16: [0..31]=r0, [32..63]=r1 */
        HVX_Vector h23 = Q6_Vh_vpack_VwVw_sat(r3, r2);
        HVX_Vector b   = Q6_Vb_vpack_VhVh_sat(h23, h01);/* i8:  [0..63]=h01, [64..127]=h23 */
        *(HVX_Vector*)(out + i) = b;
    }
    for (; i < n; i++) {
        int64_t sum = (int64_t)a[i] + (int64_t)bias[i];
        int64_t v   = sum * (int64_t)mult;
        int64_t hf  = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
        int64_t r   = (v >= 0) ? ((v + hf) >> shift) : -(((-v) + hf) >> shift);
        r += zp;
        if (r > 127) r = 127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
