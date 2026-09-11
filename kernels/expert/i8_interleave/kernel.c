/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Double-buffered: prefetch the next a/b tile (async uDMA) into VTCM while the
 * current tile is interleaved on-chip with HVX vshuff, then DMA the interleaved
 * output tile back to DDR. Staging both input streams in VTCM hides the DDR
 * latency that makes the two-stream direct-DDR baseline slow. Type-0 (1D)
 * descriptors, static/global, chained via next-ptr. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CHP 8192                 /* pairs per tile: a,b tiles 8KB each; out tile 16KB */

static desc_t d_a, d_b, d_o, pf_a, pf_b;

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    uint32_t va0=VTCM_BASE,       va1=VTCM_BASE+CHP;
    uint32_t vb0=VTCM_BASE+2*CHP,  vb1=VTCM_BASE+3*CHP;
    uint32_t vo0=VTCM_BASE+4*CHP,  vo1=VTCM_BASE+6*CHP;   /* out tile = 2*CHP */

    int nt = n / CHP;
    int rem = n - nt*CHP;
    if (nt == 0) { for (int i=0;i<n;i++){ out[2*i]=a[i]; out[2*i+1]=b[i]; } return; }

    d_a.next=(uint32_t)(uintptr_t)&d_b; d_a.ctrl=CHP; d_a.src=(uint32_t)(uintptr_t)a; d_a.dst=va0;
    d_b.next=0; d_b.ctrl=CHP; d_b.src=(uint32_t)(uintptr_t)b; d_b.dst=vb0;
    Q6_dmstart_A(&d_a); Q6_R_dmwait();

    for (int t=0; t<nt; t++) {
        uint32_t ca=(t&1)?va1:va0, cb=(t&1)?vb1:vb0, co=(t&1)?vo1:vo0;
        uint32_t na=(t&1)?va0:va1, nb=(t&1)?vb0:vb1;
        if (t+1<nt) {
            pf_a.next=(uint32_t)(uintptr_t)&pf_b; pf_a.ctrl=CHP;
            pf_a.src=(uint32_t)(uintptr_t)(a+(t+1)*CHP); pf_a.dst=na;
            pf_b.next=0; pf_b.ctrl=CHP;
            pf_b.src=(uint32_t)(uintptr_t)(b+(t+1)*CHP); pf_b.dst=nb;
            Q6_dmstart_A(&pf_a);
        }
        const HVX_Vector *va=(const HVX_Vector*)(uintptr_t)ca;
        const HVX_Vector *vb=(const HVX_Vector*)(uintptr_t)cb;
        HVX_Vector *vo=(HVX_Vector*)(uintptr_t)co;
        for (int k=0;k<CHP/128;k++) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(vb[k], va[k], -1);
            vo[2*k]=Q6_V_lo_W(p); vo[2*k+1]=Q6_V_hi_W(p);
        }
        if (t+1<nt) Q6_R_dmwait();
        d_o.next=0; d_o.ctrl=2*CHP; d_o.src=co; d_o.dst=(uint32_t)(uintptr_t)(out+t*2*CHP);
        Q6_dmstart_A(&d_o); Q6_R_dmwait();
    }
    for (int i = nt*CHP; i < nt*CHP + rem; i++) { out[2*i]=a[i]; out[2*i+1]=b[i]; }
}
