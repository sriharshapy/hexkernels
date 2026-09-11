/* EXPERT (achievability bar) = solutions/s1.c (AXPY/row-broadcast qf16
 * matmul), chosen over solutions/s2.c (per-output gather + horizontal
 * reduce) by measurement: s1 steady-state 3261 kernel cycles vs s2's 7494 --
 * the horizontal ror-shift reduce in s2 costs far more than s1's single
 * accumulate-across-d loop for this M=8,N=8,D=40 shape. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector hf_splat_bits(unsigned short bits) {
    return Q6_Vh_vsplat_R((int)bits);
}

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, hvx_hf *S,
                      int M, int N, int D, _Float16 scale) {
    static hvx_hf Kpad[40*8 + 64] HVX_ALIGN;   /* K plus a zero-padded tail */
    int total = D * N;
    for (int i = 0; i < total; i++) Kpad[i] = K[i];
    for (int i = total; i < total + 64; i++) Kpad[i] = (hvx_hf)0.0f;

    unsigned short scaleBits = *(const unsigned short *)&scale;
    HVX_Vector scaleVec = hf_splat_bits(scaleBits);

    for (int i = 0; i < M; i++) {
        unsigned short a0 = *(const unsigned short *)&Q[i*D + 0];
        HVX_Vector sa = hf_splat_bits(a0);
        HVX_Vector vb = *(const HVX_UVector *)(Kpad + 0);
        HVX_Vector acc = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
        for (int d = 1; d < D; d++) {
            unsigned short ad = *(const unsigned short *)&Q[i*D + d];
            sa = hf_splat_bits(ad);
            vb = *(const HVX_UVector *)(Kpad + d*N);
            HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(sa, vb);
            acc = Q6_Vqf16_vadd_Vqf16Vqf16(acc, prod);
        }
        HVX_Vector m = Q6_Vhf_equals_Vqf16(acc);            /* hf-round #1 */
        HVX_Vector sQf = Q6_Vqf16_vmpy_VhfVhf(m, scaleVec);
        HVX_Vector sHf = Q6_Vhf_equals_Vqf16(sQf);          /* hf-round #2 */
        const hvx_hf *rp = (const hvx_hf *)&sHf;
        for (int j = 0; j < N; j++) S[i*N + j] = rp[j];
    }
}
