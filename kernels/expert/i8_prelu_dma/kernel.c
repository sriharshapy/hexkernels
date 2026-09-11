/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Streams x[] from DDR into VTCM via uDMA, computes per-channel PReLU on the
 * on-chip copy with HVX, DMAs results back. Double-buffered: next input tile is
 * prefetched (async DMA) while the current tile computes, hiding DDR latency.
 * n_elem is a multiple of the tile size, so every tile lives in a single
 * channel and uses one alpha. */
#include <stdint.h>
#include <string.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CH 16384                 /* bytes per tile (128 vectors) */

static desc_t d_in, d_out, d_pf;

static inline HVX_Vector prelu_vec(HVX_Vector vx, HVX_Vector valpha, int shift, HVX_Vector vzero) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(vx);
    HVX_Vector res[2];
    HVX_Vector parts[2]; parts[0] = Q6_V_lo_W(xh); parts[1] = Q6_V_hi_W(xh);
    for (int k = 0; k < 2; k++) {
        HVX_Vector prod    = Q6_Vh_vmpyi_VhVh(parts[k], valpha);
        HVX_Vector shifted = Q6_Vh_vasr_VhR(prod, shift);
        HVX_VectorPred pos = Q6_Q_vcmp_gt_VhVh(parts[k], vzero);
        res[k] = Q6_V_vmux_QVV(pos, parts[k], shifted);
    }
    return Q6_Vb_vasr_VhVhR_sat(res[1], res[0], 0);
}

void candidate_kernel(const int8_t *x, int8_t *out,
                      int n_ch, int n_elem,
                      const int8_t *alpha, int shift) {
    HVX_Vector vzero = Q6_V_vzero();
    const uint32_t vt_x0 = VTCM_BASE,        vt_x1 = VTCM_BASE + CH;
    const uint32_t vt_o0 = VTCM_BASE + 2*CH, vt_o1 = VTCM_BASE + 3*CH;

    int tiles_per_ch = n_elem / CH;
    int ntiles = n_ch * tiles_per_ch;
    if (tiles_per_ch == 0) {
        /* fallback: scalar */
        for (int c = 0; c < n_ch; c++) { int a=(int)alpha[c];
            for (int i=0;i<n_elem;i++){ int8_t v=x[c*n_elem+i]; if(v>0){out[c*n_elem+i]=v;continue;}
                int r=((int)v*a)>>shift; if(r>127)r=127; if(r<-128)r=-128; out[c*n_elem+i]=(int8_t)r; } }
        return;
    }

    /* Prologue: DMA tile 0 into slot 0. */
    d_in.next = 0; d_in.ctrl = CH; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int c = 0; c < ntiles; c++) {
        uint32_t cur_x = (c & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (c & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (c & 1) ? vt_x0 : vt_x1;

        if (c + 1 < ntiles) {
            const int8_t *nx = x + (c + 1) * CH;
            d_pf.next = 0; d_pf.ctrl = CH; d_pf.src = (uint32_t)(uintptr_t)nx; d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        int ch = c / tiles_per_ch;
        int a = (int)alpha[ch];
        uint32_t aw = ((uint32_t)(a & 0xFFFF) << 16) | (a & 0xFFFF);
        HVX_Vector valpha = Q6_V_vsplat_R(aw);

        HVX_Vector *px = (HVX_Vector *)(uintptr_t)cur_x;
        HVX_Vector *po = (HVX_Vector *)(uintptr_t)cur_o;
        for (int v = 0; v < CH / 128; v++) po[v] = prelu_vec(px[v], valpha, shift, vzero);

        if (c + 1 < ntiles) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CH; d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + c * CH);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
}
