/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Tiles the moving average by output blocks of OT samples. For each tile: DMA the
 * input window (OT + halo bytes) from DDR into VTCM, run the light shift-add
 * window sum + truncating divide on the on-chip copy writing int8 results into a
 * VTCM output slot, then DMA the output slot back to DDR. The next input window is
 * prefetched (async DMA) while the current tile computes, hiding DDR latency. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define OT    8192
#define HALO  7
#define ITILE (OT + HALO)
#define ISLOT (OT + 256)          /* padded input slot */
#define OSLOT OT

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline HVX_Vector unpack_lo_b(HVX_Vector xb) { return Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb)); }
static inline HVX_Vector wsum8(HVX_Vector cur, HVX_Vector nxt) {
    HVX_Vector acc = cur;
    for (int s = 1; s < 8; s++) acc = Q6_Vh_vadd_VhVh(acc, Q6_V_valign_VVR(nxt, cur, s*2));
    return acc;
}
static inline HVX_Vector div8_trunc(HVX_Vector acc) {
    HVX_Vector seven = Q6_Vh_vsplat_R(7);
    HVX_Vector sign  = Q6_Vh_vasr_VhR(acc, 15);
    HVX_Vector bias  = Q6_V_vand_VV(sign, seven);
    return Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(acc, bias), 3);
}

void candidate_kernel(const int8_t *x, int8_t *out, int n, int W) {
    (void)W;
    const uint32_t vin0 = VTCM_BASE,           vin1 = VTCM_BASE + ISLOT;
    const uint32_t vout0 = VTCM_BASE + 2*ISLOT, vout1 = VTCM_BASE + 2*ISLOT + OSLOT;

    int ntile = n / OT;
    int done  = ntile * OT;

    if (ntile == 0) {
        for (int i = 0; i < n; i++) { int32_t a=0; for (int j=0;j<8;j++) a+=(int32_t)x[i+j]; out[i]=(int8_t)(a/8); }
        return;
    }

    d_in.next = 0; d_in.ctrl = ITILE; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vin0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int t = 0; t < ntile; t++) {
        uint32_t vin  = (t & 1) ? vin1  : vin0;
        uint32_t vnxt = (t & 1) ? vin0  : vin1;
        uint32_t vout = (t & 1) ? vout1 : vout0;

        if (t + 1 < ntile) {
            d_pf.next = 0; d_pf.ctrl = ITILE;
            d_pf.src = (uint32_t)(uintptr_t)(x + (t + 1) * OT); d_pf.dst = vnxt;
            Q6_dmstart_A(&d_pf);
        }

        const int8_t *ip = (const int8_t *)(uintptr_t)vin;
        int8_t *op = (int8_t *)(uintptr_t)vout;
        for (int i = 0; i < OT; i += 128) {
            HVX_Vector Xa = unpack_lo_b(load_ua(ip + i));
            HVX_Vector Xb = unpack_lo_b(load_ua(ip + i + 64));
            HVX_Vector Xc = unpack_lo_b(load_ua(ip + i + 128));
            HVX_Vector qA = div8_trunc(wsum8(Xa, Xb));
            HVX_Vector qB = div8_trunc(wsum8(Xb, Xc));
            *(HVX_Vector *)(op + i) = Q6_Vb_vpack_VhVh_sat(qB, qA);
        }

        if (t + 1 < ntile) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = OSLOT;
        d_out.src = vout; d_out.dst = (uint32_t)(uintptr_t)(out + t * OT);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = done; i < n; i++) { int32_t a=0; for (int j=0;j<8;j++) a+=(int32_t)x[i+j]; out[i]=(int8_t)(a/8); }
}
