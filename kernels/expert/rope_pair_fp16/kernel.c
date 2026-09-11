/*
 * HVX RoPE-on-pairs, qf16 pipeline. n_pairs=100 fits 2 HVX vectors (64
 * fp16 lanes each). Deinterleave x into contiguous re[]/im[] (scalar --
 * cheap, memory-only), then do the 4 multiplies + add/sub in qf16 across
 * 2 blocks, then re-interleave into out[] (scalar).
 */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define NBUF 128   /* 2 HVX vectors of 64 fp16 lanes; covers n_pairs<=128 */

void candidate_kernel(const hvx_hf *x, const hvx_hf *cos, const hvx_hf *sin,
                      hvx_hf *out, int n_pairs)
{
    static hvx_hf re[NBUF]   HVX_ALIGN;
    static hvx_hf im[NBUF]   HVX_ALIGN;
    static hvx_hf cosb[NBUF] HVX_ALIGN;
    static hvx_hf sinb[NBUF] HVX_ALIGN;
    static hvx_hf outre[NBUF] HVX_ALIGN;
    static hvx_hf outim[NBUF] HVX_ALIGN;

    for (int p = 0; p < n_pairs; p++) {
        re[p] = x[2*p];
        im[p] = x[2*p + 1];
        cosb[p] = cos[p];
        sinb[p] = sin[p];
    }
    for (int p = n_pairs; p < NBUF; p++) {
        re[p] = im[p] = cosb[p] = sinb[p] = (hvx_hf)0.0f;
    }

    for (int off = 0; off < NBUF; off += 64) {
        HVX_Vector vre  = *(const HVX_Vector *)(re + off);
        HVX_Vector vim  = *(const HVX_Vector *)(im + off);
        HVX_Vector vcos = *(const HVX_Vector *)(cosb + off);
        HVX_Vector vsin = *(const HVX_Vector *)(sinb + off);

        HVX_Vector p1 = Q6_Vqf16_vmpy_VhfVhf(vre, vcos);
        HVX_Vector p2 = Q6_Vqf16_vmpy_VhfVhf(vim, vsin);
        HVX_Vector outReQf = Q6_Vqf16_vsub_Vqf16Vqf16(p1, p2);

        HVX_Vector p3 = Q6_Vqf16_vmpy_VhfVhf(vre, vsin);
        HVX_Vector p4 = Q6_Vqf16_vmpy_VhfVhf(vim, vcos);
        HVX_Vector outImQf = Q6_Vqf16_vadd_Vqf16Vqf16(p3, p4);

        *(HVX_Vector *)(outre + off) = Q6_Vhf_equals_Vqf16(outReQf);
        *(HVX_Vector *)(outim + off) = Q6_Vhf_equals_Vqf16(outImQf);
    }

    for (int p = 0; p < n_pairs; p++) {
        out[2*p]     = outre[p];
        out[2*p + 1] = outim[p];
    }
}
