/* EXPERT (achievability bar) -- hvx + dma + vtcm.
 * Tiles the 5-tap FIR by output blocks of OT samples. For each tile: DMA the
 * input window (OT + ntaps-1 halo bytes) from DDR into VTCM, run the FIR+
 * requant on the on-chip copy writing int8 results into a VTCM output slot,
 * then DMA the output slot back to DDR. The next input window is prefetched
 * (async DMA) while the current tile computes, hiding DDR latency behind
 * compute. Both the input reads and the output writes hit VTCM instead of
 * DDR during the compute phase. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define OT    8192                /* outputs per tile */
#define HALO  4                   /* ntaps-1 (ntaps fixed to 5) */
#define ITILE (OT + HALO)         /* input bytes per tile */
#define ISLOT 8448                /* input slot stride, padded/aligned */
#define OSLOT OT                  /* output slot bytes */

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline HVX_Vector fir5_64(const int8_t *xp, const int8_t *taps, int ntaps) {
    HVX_Vector acc = Q6_V_vzero();
    for (int j = 0; j < ntaps; j++) {
        HVX_Vector xb = load_ua(xp + j);
        HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
        HVX_Vector tv = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[j]);
        acc = Q6_Vh_vadd_VhVh(acc, Q6_Vh_vmpyi_VhVh(xh, tv));
    }
    return acc;
}
static inline HVX_Vector requant5(HVX_Vector acc, int shift) {
    HVX_Vector vzero = Q6_V_vzero();
    int hv = shift > 0 ? (1 << (shift - 1)) : 0;
    HVX_Vector vhalf = Q6_Vh_vsplat_R(hv);
    HVX_Vector absacc = Q6_Vh_vabs_Vh(acc);
    HVX_Vector sh = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(absacc, vhalf), shift);
    HVX_VectorPred neg = Q6_Q_vcmp_gt_VhVh(vzero, acc);
    return Q6_V_vmux_QVV(neg, Q6_Vh_vsub_VhVh(vzero, sh), sh);
}
static inline int8_t fir5_ref(const int8_t *xp, const int8_t *taps, int ntaps, int shift) {
    int32_t acc = 0;
    for (int j = 0; j < ntaps; j++) acc += (int32_t)xp[j] * (int32_t)taps[j];
    int32_t half = shift > 0 ? (1 << (shift - 1)) : 0;
    int32_t r = (acc >= 0) ? ((acc + half) >> shift) : -(((-acc) + half) >> shift);
    if (r > 127) r = 127; if (r < -128) r = -128;
    return (int8_t)r;
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int8_t *out,
                      int n, int ntaps, int shift) {
    const uint32_t vin0 = VTCM_BASE,             vin1 = VTCM_BASE + ISLOT;
    const uint32_t vout0 = VTCM_BASE + 2*ISLOT,  vout1 = VTCM_BASE + 2*ISLOT + OSLOT;

    int ntile = n / OT;
    int done  = ntile * OT;

    if (ntile == 0) {
        for (int i = 0; i < n; i++) out[i] = fir5_ref(x + i, taps, ntaps, shift);
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
            HVX_Vector lo = fir5_64(ip + i,      taps, ntaps);
            HVX_Vector hi = fir5_64(ip + i + 64, taps, ntaps);
            HVX_Vector rlo = requant5(lo, shift);
            HVX_Vector rhi = requant5(hi, shift);
            *(HVX_Vector *)(op + i) = Q6_Vb_vpack_VhVh_sat(rhi, rlo);
        }

        if (t + 1 < ntile) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = OSLOT;
        d_out.src = vout; d_out.dst = (uint32_t)(uintptr_t)(out + t * OT);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    for (int i = done; i < n; i++) out[i] = fir5_ref(x + i, taps, ntaps, shift);
}
