/* EXPERT -- HVX-vectorized flash-attention tile. Per query row, streams the
 * SK/FLASH_TILE_K key/value tiles maintaining the online-softmax recurrence:
 *  - QK^T dot: one qf16 vector multiply (DH=64 = one full 128B fp16 vector,
 *    no padding) + a 6-step vror/vqf16-add horizontal reduction (same
 *    halving-reduction shape as the classic int32 hreduce, just 2-byte lanes).
 *  - weighted-V accumulate + old-state rescale: the SAME proven "vector-per-
 *    key" qf16 accumulate chain as online_softmax_step_fp16's expert (splat
 *    the scalar softmax weight / rescale factor, Q6_Vqf16_vmpy + running
 *    Q6_Vqf16_vadd_Vqf16Vqf16), carried hf-resident ACROSS tiles.
 *  - final normalize: one vector multiply by the splatted 1/l instead of a
 *    64-wide scalar divide loop.
 * exp() itself stays scalar (HVX has no vector transcendental). */
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

/* Dot of two 64-lane fp16 vectors -> float, via qf16 multiply + 6-step
 * halving vror/vadd horizontal reduction (128B -> 64 -> 32 -> 16 -> 8 -> 4 -> 2). */
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

            float p[FLASH_TILE_K], psum = 0.0f;
            for (int k = 0; k < FLASH_TILE_K; k++) {
                p[k] = expf(sc[k] - new_m);
                psum += p[k];
            }
            l = l * corr + psum;

            HVX_Vector accv    = *(const HVX_Vector *)abuf;
            HVX_Vector corrVec = hf_splat_f(corr);
            HVX_Vector accQf   = Q6_Vqf16_vmpy_VhfVhf(accv, corrVec);   /* rescale OLD state */
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

        /* Final normalize: one vector multiply by the splatted 1/l. */
        HVX_Vector accv  = *(const HVX_Vector *)abuf;
        HVX_Vector invlV = hf_splat_f(1.0f / l);
        HVX_Vector outQf = Q6_Vqf16_vmpy_VhfVhf(accv, invlV);
        *(HVX_Vector *)(O + (size_t)i * DH) = Q6_Vhf_equals_Vqf16(outQf);
    }
}
