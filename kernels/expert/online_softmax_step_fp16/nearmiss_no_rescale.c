/* Near-miss: THE classic online-softmax bug. Computes m_new, corr, p[j],
 * l_new correctly, but forgets to rescale the OLD accumulator (acc) and
 * the OLD running_sum by corr before folding in the new block's
 * contribution -- i.e. treats corr as always 1.0 for the update itself
 * (even though it was computed). Compiles, uses HVX, and is correct
 * whenever m_new==m_old (first block, corr=1 exactly) but silently wrong
 * whenever the block max exceeds the running max (corr<1, a real
 * rescale needed) -- exactly the edge case the harness's second test
 * exercises. */
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

    float m_blk = (float)scores[0];
    for (int j = 1; j < BLK; j++) {
        float v = (float)scores[j];
        if (v > m_blk) m_blk = v;
    }
    float m_new = (m_old > m_blk) ? m_old : m_blk;
    float corr = expf(m_old - m_new);   /* computed... */
    (void)corr;                          /* ...but never applied below (the bug) */

    float p[24];
    float psum = 0.0f;
    for (int j = 0; j < BLK; j++) {
        p[j] = expf((float)scores[j] - m_new);
        psum += p[j];
    }
    /* BUG: l_new should be l_old*corr + psum -- drops the *corr rescale. */
    float l_new = l_old + psum;

    static hvx_hf vbuf[64] HVX_ALIGN;
    static hvx_hf abuf[64] HVX_ALIGN;
    for (int d = 0; d < D; d++) abuf[d] = acc[d];
    for (int d = D; d < 64; d++) abuf[d] = (hvx_hf)0.0f;
    HVX_Vector accv = *(const HVX_Vector *)abuf;
    /* BUG: no corr multiply on the old accumulator -- fed straight into
     * the qf16 chain unscaled. */
    HVX_Vector accQf = Q6_Vqf16_vadd_VhfVhf(accv, hf_splat_f(0.0f));

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
