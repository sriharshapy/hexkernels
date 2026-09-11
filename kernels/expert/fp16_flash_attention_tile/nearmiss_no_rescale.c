/* NEARMISS -- THE classic flash-attention bug: computes corr = exp(m_old-m_new)
 * but never applies it to the OLD running sum / accumulator before folding in
 * the new tile's contribution (i.e. behaves as if corr were always 1.0).
 * Compiles, uses HVX, and is correct only on the tile where the running max
 * doesn't change (rare); wrong on every row where a later tile raises the
 * running max above an earlier tile's max (the harness's row-0 edge case
 * pins exactly this: the dominant key is in the LAST tile). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

static inline HVX_Vector hf_splat_f(float f) {
    hvx_hf v = (hvx_hf)f;
    unsigned short bits = *(const unsigned short *)&v;
    return Q6_Vh_vsplat_R((int)bits);
}
static inline float hf_dot64(HVX_Vector a, HVX_Vector b) {
    HVX_Vector acc = Q6_Vqf16_vmpy_VhfVhf(a, b);
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 64));
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 32));
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 16));
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 8));
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 4));
    acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, Q6_V_vror_VR(acc, 2));
    HVX_Vector hf = Q6_Vhf_equals_Vqf16(acc);
    const hvx_hf *p = (const hvx_hf *)&hf;
    return (float)p[0];
}

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, const hvx_hf *V,
                      hvx_hf *O, int SQ, int SK, int DH) {
    const int ntiles = SK / FLASH_TILE_K;
    static hvx_hf abuf[64] HVX_ALIGN;

    for (int i = 0; i < SQ; i++) {
        HVX_Vector qv = *(const HVX_Vector *)(Q + (size_t)i * DH);
        float m = -1e30f, l = 0.0f;
        for (int d = 0; d < DH; d++) abuf[d] = (hvx_hf)0.0f;

        for (int t = 0; t < ntiles; t++) {
            float sc[FLASH_TILE_K];
            float tile_max = -1e30f;
            for (int k = 0; k < FLASH_TILE_K; k++) {
                HVX_Vector kv = *(const HVX_Vector *)(K + (size_t)(t*FLASH_TILE_K + k) * DH);
                float raw = hf_dot64(qv, kv);
                sc[k] = raw * FLASH_SCALE;
                if (sc[k] > tile_max) tile_max = sc[k];
            }
            float new_m = (m > tile_max) ? m : tile_max;
            float corr = expf(m - new_m);
            (void)corr;   /* BUG: computed but never applied below */

            float p[FLASH_TILE_K], psum = 0.0f;
            for (int k = 0; k < FLASH_TILE_K; k++) {
                p[k] = expf(sc[k] - new_m);
                psum += p[k];
            }
            l = l + psum;   /* BUG: should be l*corr + psum */

            HVX_Vector accv  = *(const HVX_Vector *)abuf;
            /* BUG: old accumulator fed in unscaled (no *corr). */
            HVX_Vector accQf = Q6_Vqf16_vadd_VhfVhf(accv, hf_splat_f(0.0f));
            for (int k = 0; k < FLASH_TILE_K; k++) {
                HVX_Vector vv   = *(const HVX_Vector *)(V + (size_t)(t*FLASH_TILE_K + k) * DH);
                HVX_Vector pVec = hf_splat_f(p[k]);
                HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(vv, pVec);
                accQf = Q6_Vqf16_vadd_Vqf16Vqf16(accQf, prod);
            }
            HVX_Vector accHf = Q6_Vhf_equals_Vqf16(accQf);
            *(HVX_Vector *)abuf = accHf;

            m = new_m;
        }

        HVX_Vector accv  = *(const HVX_Vector *)abuf;
        HVX_Vector invlV = hf_splat_f(1.0f / l);
        HVX_Vector outQf = Q6_Vqf16_vmpy_VhfVhf(accv, invlV);
        *(HVX_Vector *)(O + (size_t)i * DH) = Q6_Vhf_equals_Vqf16(outQf);
    }
}
