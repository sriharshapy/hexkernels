#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
#define CH_EL 4096
static desc_t d_in, d_out;
void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    const uint32_t vt_x = VTCM_BASE, vt_o = VTCM_BASE + CH_EL;
    HVX_Vector vzero = Q6_V_vzero();
    int nfull = n / CH_EL, rem = n - nfull * CH_EL;
    if (nfull == 0) { for (int i = 0; i < n; i++) out[i] = (x[i] < 0) ? 0 : x[i]; return; }
    const int nbv = CH_EL / 128;                 /* 128B byte-vectors per tile */
    for (int c = 0; c < nfull; c++) {
        d_in.next = 0; d_in.ctrl = CH_EL; d_in.src = (uint32_t)(uintptr_t)(x + c*CH_EL); d_in.dst = vt_x;
        Q6_dmstart_A(&d_in); Q6_R_dmwait();       /* single-buffer: fetch, then compute */
        HVX_Vector *px = (HVX_Vector *)(uintptr_t)vt_x, *po = (HVX_Vector *)(uintptr_t)vt_o;
        for (int vb = 0; vb < nbv; vb++) po[vb] = Q6_Vb_vmax_VbVb(px[vb], vzero);
        d_out.next = 0; d_out.ctrl = CH_EL; d_out.src = vt_o; d_out.dst = (uint32_t)(uintptr_t)(out + c*CH_EL);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
    for (int i = nfull*CH_EL; i < nfull*CH_EL + rem; i++) out[i] = (x[i] < 0) ? 0 : x[i];
}
