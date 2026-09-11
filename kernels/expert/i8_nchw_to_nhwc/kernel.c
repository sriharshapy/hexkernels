/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Double-buffered: prefetch the next tile of all four channel planes (async
 * uDMA, 4 chained Type-0 descriptors) into VTCM while the current tile is
 * interleaved on-chip with two levels of HVX vshuff, then DMA the interleaved
 * NHWC output tile back to DDR. Staging the four plane streams in VTCM hides the
 * DDR latency that makes the four-stream direct-DDR baseline slow. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define CHP 4096                  /* per-plane bytes per tile (32 blocks) */

static desc_t di[4], pf[4], d_o;

/* build a chain of 4 plane-load descriptors from DDR base `src` (plane stride HW)
 * into 4 VTCM slots at `dstv` (slot stride CHP). */
static void build_in(desc_t *d, const int8_t *src, int off, int HW, uint32_t dstv) {
    for (int c=0;c<4;c++) {
        d[c].next = (c<3)?(uint32_t)(uintptr_t)&d[c+1]:0;
        d[c].ctrl = CHP;
        d[c].src  = (uint32_t)(uintptr_t)(src + c*HW + off);
        d[c].dst  = dstv + c*CHP;
    }
}

void candidate_kernel(const int8_t *in, int8_t *out, int C, int HW) {
    (void)C;
    uint32_t s0 = VTCM_BASE;            /* buffer A: 4 planes -> 4*CHP */
    uint32_t s1 = VTCM_BASE + 4*CHP;    /* buffer B */
    uint32_t o0 = VTCM_BASE + 8*CHP;    /* out tile A: 4*CHP */
    uint32_t o1 = VTCM_BASE + 12*CHP;   /* out tile B */

    int nt = HW / CHP;
    int rem = HW - nt*CHP;
    if (nt == 0) {
        for (int p=0;p<HW;p++) for(int c=0;c<4;c++) out[p*4+c]=in[c*HW+p];
        return;
    }
    build_in(di, in, 0, HW, s0);
    Q6_dmstart_A(&di[0]); Q6_R_dmwait();

    for (int t=0;t<nt;t++) {
        uint32_t cs=(t&1)?s1:s0, co=(t&1)?o1:o0, ns=(t&1)?s0:s1;
        if (t+1<nt) { build_in(pf, in, (t+1)*CHP, HW, ns); Q6_dmstart_A(&pf[0]); }

        const HVX_Vector *p0=(const HVX_Vector*)(uintptr_t)(cs+0*CHP);
        const HVX_Vector *p1=(const HVX_Vector*)(uintptr_t)(cs+1*CHP);
        const HVX_Vector *p2=(const HVX_Vector*)(uintptr_t)(cs+2*CHP);
        const HVX_Vector *p3=(const HVX_Vector*)(uintptr_t)(cs+3*CHP);
        HVX_Vector *vo=(HVX_Vector*)(uintptr_t)co;
        for (int k=0;k<CHP/128;k++) {
            HVX_Vector v0=p0[k],v1=p1[k],v2=p2[k],v3=p3[k];
            HVX_VectorPair a=Q6_W_vshuff_VVR(v1,v0,-1), b=Q6_W_vshuff_VVR(v3,v2,-1);
            HVX_VectorPair c=Q6_W_vshuff_VVR(Q6_V_lo_W(b),Q6_V_lo_W(a),-2);
            HVX_VectorPair d=Q6_W_vshuff_VVR(Q6_V_hi_W(b),Q6_V_hi_W(a),-2);
            vo[4*k+0]=Q6_V_lo_W(c); vo[4*k+1]=Q6_V_hi_W(c);
            vo[4*k+2]=Q6_V_lo_W(d); vo[4*k+3]=Q6_V_hi_W(d);
        }
        if (t+1<nt) Q6_R_dmwait();
        d_o.next=0; d_o.ctrl=4*CHP; d_o.src=co; d_o.dst=(uint32_t)(uintptr_t)(out + t*4*CHP);
        Q6_dmstart_A(&d_o); Q6_R_dmwait();
    }
    for (int p=nt*CHP; p<nt*CHP+rem; p++)
        for (int c=0;c<4;c++) out[p*4+c]=in[c*HW+p];
}
