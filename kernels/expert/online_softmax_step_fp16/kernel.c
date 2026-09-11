/*
 * HVX online-softmax update: "vector-per-key" accumulate. Since D=48 <= 64
 * fp16 lanes, the ENTIRE weighted-value accumulation for one key fits in a
 * single HVX vector op. Loop over the BLK=24 keys, broadcasting each
 * scalar softmax weight p[j] and multiply-accumulating the full V row (one
 * vrow-wide vector op replaces a D-wide scalar loop, x BLK times). The
 * OLD accumulator's corr-rescale is folded into the SAME qf16 chain (its
 * "product" is corr*acc, added first). exp itself stays scalar (HVX has
 * no vector transcendental).
 */
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

void candidate_kernel(const hvx_hf *scores, const hvx_hf *V,
                      hvx_hf *running_max, hvx_hf *running_sum, hvx_hf *acc,
                      int BLK, int D)
{
    float m_old = (float)*running_max;
    float l_old = (float)*running_sum;

    /* Block max: scalar (BLK is small; the exp loop below dominates and is
     * unavoidably scalar anyway). */
    float m_blk = (float)scores[0];
    for (int j = 1; j < BLK; j++) {
        float v = (float)scores[j];
        if (v > m_blk) m_blk = v;
    }
    float m_new = (m_old > m_blk) ? m_old : m_blk;
    float corr = expf(m_old - m_new);

    float p[24];
    float psum = 0.0f;
    for (int j = 0; j < BLK; j++) {
        p[j] = expf((float)scores[j] - m_new);
        psum += p[j];
    }
    float l_new = l_old * corr + psum;

    /* Vectorized accumulate over D (<=64 lanes -> ONE HVX vector):
     *   acc_new = acc*corr + sum_j p[j] * V[j,:]
     */
    static hvx_hf vbuf[64] HVX_ALIGN;
    static hvx_hf abuf[64] HVX_ALIGN;
    for (int d = 0; d < D; d++) abuf[d] = acc[d];
    for (int d = D; d < 64; d++) abuf[d] = (hvx_hf)0.0f;
    HVX_Vector accv    = *(const HVX_Vector *)abuf;
    HVX_Vector corrVec = hf_splat_f(corr);
    HVX_Vector accQf   = Q6_Vqf16_vmpy_VhfVhf(accv, corrVec);   /* acc*corr, qf16 */

    for (int j = 0; j < BLK; j++) {
        const hvx_hf *Vrow = V + (size_t)j * D;
        for (int d = 0; d < D; d++) vbuf[d] = Vrow[d];
        for (int d = D; d < 64; d++) vbuf[d] = (hvx_hf)0.0f;
        HVX_Vector vv   = *(const HVX_Vector *)vbuf;
        HVX_Vector pVec = hf_splat_f(p[j]);
        HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(vv, pVec);
        accQf = Q6_Vqf16_vadd_Vqf16Vqf16(accQf, prod);
    }

    HVX_Vector accHf = Q6_Vhf_equals_Vqf16(accQf);
    const hvx_hf *outp = (const hvx_hf *)&accHf;
    for (int d = 0; d < D; d++) acc[d] = outp[d];

    *running_max = (hvx_hf)m_new;
    *running_sum = (hvx_hf)l_new;
}
