/* EXPERT (achievability bar) — HVX-vectorized sparse (banded) attention.
 * SEQ=8, HEAD_DIM=16. The AV stage dominates the requant cost (SEQ*HEAD_DIM=128
 * saturating requants per invocation); we vectorize it 16-wide: for each query
 * row i, acc[0:16] = sum_j prob[j]*V[j,0:16] accumulated in 16 int32 lanes, then
 * one vectorized sign-aware requant (amult>0) + saturating pack -> 16 int8. The
 * banded QK^T scores and the scalar softmax (max / exp-LUT / integer divide,
 * only SEQ=8 wide and not SIMD-friendly) stay scalar. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define H        1
#define SEQ      8
#define HEAD_DIM 16
#define MASK_VAL ((int8_t)(-128))

static int8_t requant_i8(int32_t raw, int32_t mult, int shift) {
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (v >= 0) ? ((v + half) >> shift) : -((-v + half) >> shift);
    if (q >  127) q =  127;
    if (q < -128) q = -128;
    return (int8_t)q;
}

/* Sign-aware vector requant of 16..32 int32 lanes (mult>0), no zp. */
static inline HVX_Vector requant_vec(HVX_Vector v, int32_t mult, int shift) {
    int half = shift > 0 ? (1 << (shift - 1)) : 0;
    HVX_Vector vmult = Q6_V_vsplat_R((uint32_t)(uint16_t)mult);
    HVX_Vector vhalf = Q6_V_vsplat_R((uint32_t)half);
    HVX_Vector sm    = Q6_Vw_vasr_VwR(v, 31);
    HVX_Vector av    = Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(v, sm), sm);
    HVX_Vector am    = Q6_Vw_vmpyie_VwVuh(av, vmult);
    HVX_Vector sh    = Q6_Vw_vasr_VwR(Q6_Vw_vadd_VwVw(am, vhalf), shift);
    return Q6_Vw_vsub_VwVw(Q6_V_vxor_VV(sh, sm), sm);
}

void candidate_kernel(const int8_t *Q, const int8_t *K, const int8_t *V,
                      const uint8_t *exp_lut, int8_t *out,
                      int window, int32_t smult, int sshift,
                      int32_t amult, int ashift) {
    int8_t obuf[128] __attribute__((aligned(128)));
    for (int i = 0; i < SEQ; i++) {
        /* --- banded QK^T scores (scalar) --- */
        int8_t scores[SEQ];
        for (int j = 0; j < SEQ; j++) {
            int dist = i - j; if (dist < 0) dist = -dist;
            if (dist <= window) {
                int32_t acc = 0;
                for (int d = 0; d < HEAD_DIM; d++)
                    acc += (int32_t)Q[i*HEAD_DIM+d] * (int32_t)K[j*HEAD_DIM+d];
                scores[j] = requant_i8(acc, smult, sshift);
            } else {
                scores[j] = MASK_VAL;
            }
        }
        /* --- softmax (scalar) --- */
        int8_t m = scores[0];
        for (int j = 1; j < SEQ; j++) if (scores[j] > m) m = scores[j];
        int32_t e[SEQ], S = 0;
        for (int j = 0; j < SEQ; j++) {
            int diff = (int)scores[j] - (int)m;
            if (diff < -255) diff = -255;
            e[j] = (int32_t)exp_lut[diff + 255];
            S += e[j];
        }
        int32_t half_S = S / 2;
        uint32_t prob[SEQ];
        for (int j = 0; j < SEQ; j++)
            prob[j] = (uint32_t)((e[j] * 255 + half_S) / S);

        /* --- AV (vectorized 16-wide) --- */
        HVX_Vector acc = Q6_V_vzero();
        for (int j = 0; j < SEQ; j++) {
            HVX_Vector vb = *(const HVX_UVector *)(V + j*HEAD_DIM);   /* first 16 bytes = V row j */
            HVX_Vector vh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(vb));         /* int8->int16 (lanes 0..15 valid) */
            HVX_Vector vw = Q6_V_lo_W(Q6_Ww_vunpack_Vh(vh));         /* int16->int32 (lanes 0..15) */
            HVX_Vector vp = Q6_V_vsplat_R(prob[j]);                  /* prob[j] (<=255) unsigned */
            acc = Q6_Vw_vadd_VwVw(acc, Q6_Vw_vmpyie_VwVuh(vw, vp));
        }
        HVX_Vector r   = requant_vec(acc, amult, ashift);
        HVX_Vector r16 = Q6_Vh_vpack_VwVw_sat(r, r);
        HVX_Vector r8  = Q6_Vb_vpack_VhVh_sat(r16, r16);
        *(HVX_Vector *)obuf = r8;
        for (int d = 0; d < HEAD_DIM; d++) out[i*HEAD_DIM+d] = obuf[d];
    }
}
