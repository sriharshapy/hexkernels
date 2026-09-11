#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL 4096
static desc_t d_in_a, d_in_b, d_out;
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const uint32_t vt_a = VTCM_BASE, vt_b = VTCM_BASE + CH_EL, vt_o = VTCM_BASE + 2*CH_EL;
    int nfull = n / CH_EL, rem = n - nfull * CH_EL;
    if (nfull == 0) {
        for (int i = 0; i < n; i++) {
            int s = (int)a[i] + (int)b[i];
            if (s > 127) s = 127;
            if (s < -128) s = -128;
            out[i] = (int8_t)s;
        }
        return;
    }
    const int nbv = CH_EL / 128;                 /* 128B byte-vectors per tile */
    for (int c = 0; c < nfull; c++) {
        d_in_a.next = 0; d_in_a.ctrl = CH_EL; d_in_a.src = (uint32_t)(uintptr_t)(a + c*CH_EL); d_in_a.dst = vt_a;
        Q6_dmstart_A(&d_in_a); Q6_R_dmwait();     /* single-buffer: fetch a, then b, then compute */
        d_in_b.next = 0; d_in_b.ctrl = CH_EL; d_in_b.src = (uint32_t)(uintptr_t)(b + c*CH_EL); d_in_b.dst = vt_b;
        Q6_dmstart_A(&d_in_b); Q6_R_dmwait();
        HVX_Vector *pa = (HVX_Vector *)(uintptr_t)vt_a, *pb = (HVX_Vector *)(uintptr_t)vt_b, *po = (HVX_Vector *)(uintptr_t)vt_o;
        for (int vb = 0; vb < nbv; vb++) po[vb] = Q6_Vb_vadd_VbVb_sat(pa[vb], pb[vb]);
        d_out.next = 0; d_out.ctrl = CH_EL; d_out.src = vt_o; d_out.dst = (uint32_t)(uintptr_t)(out + c*CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
    for (int i = nfull*CH_EL; i < nfull*CH_EL + rem; i++) {
        int s = (int)a[i] + (int)b[i];
        if (s > 127) s = 127;
        if (s < -128) s = -128;
        out[i] = (int8_t)s;
    }
}
