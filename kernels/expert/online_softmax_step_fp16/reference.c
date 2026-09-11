/* PLAIN SCALAR online-softmax update -- the DENOMINATOR baseline. No HVX
 * intrinsics, no vector types anywhere. */
#include "kernel_api.h"
#include <math.h>

void candidate_kernel(const hvx_hf *scores, const hvx_hf *V,
                      hvx_hf *running_max, hvx_hf *running_sum, hvx_hf *acc,
                      int BLK, int D) {
    float m_old = (float)*running_max;
    float l_old = (float)*running_sum;

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

    for (int d = 0; d < D; d++) {
        float a = (float)acc[d] * corr;
        for (int j = 0; j < BLK; j++) {
            a += p[j] * (float)V[j*D + d];
        }
        acc[d] = (hvx_hf)a;
    }

    *running_max = (hvx_hf)m_new;
    *running_sum = (hvx_hf)l_new;
}
