/* EXPERT (achievability bar) = solutions/s1.c (per-output dot product via a
 * single qf16 vmpy + ror-shift horizontal reduce), chosen over
 * solutions/s2.c (same vmpy, but an extract-and-sum SCALAR horizontal
 * reduce) by measurement: s1 steady-state 5466 kernel cycles vs s2's 80246
 * -- summing 64 lanes with a scalar C loop per output element is far more
 * expensive than the vector ror-shift butterfly reduce. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline HVX_Vector hreduce_qf16(HVX_Vector v) {
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vqf16_vadd_Vqf16Vqf16(v, Q6_V_vror_VR(v, 2));
    return v;
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *V, hvx_hf *O,
                      int M, int N, int D)
{
    static hvx_hf vrows[8][64] HVX_ALIGN;
    static hvx_hf vrow_a[64] HVX_ALIGN;

    for (int d = 0; d < D; d++) {
        for (int j = 0; j < N; j++) vrows[d][j] = V[d*N + j];
        for (int j = N; j < 64; j++) vrows[d][j] = (hvx_hf)0.0f;
    }

    for (int i = 0; i < M; i++) {
        const hvx_hf *Arow = A + i * N;
        for (int j = 0; j < N; j++) vrow_a[j] = Arow[j];
        for (int j = N; j < 64; j++) vrow_a[j] = (hvx_hf)0.0f;
        HVX_Vector va = *(const HVX_Vector *)vrow_a;

        hvx_hf *Orow = O + i * D;
        for (int d = 0; d < D; d++) {
            HVX_Vector vb = *(const HVX_Vector *)vrows[d];
            HVX_Vector prod = Q6_Vqf16_vmpy_VhfVhf(va, vb);
            HVX_Vector red = hreduce_qf16(prod);
            HVX_Vector sHf = Q6_Vhf_equals_Vqf16(red);
            const hvx_hf *rp = (const hvx_hf *)&sHf;
            Orow[d] = rp[0];
        }
    }
}
