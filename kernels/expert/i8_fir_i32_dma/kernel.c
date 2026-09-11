/* EXPERT (achievability bar) — hvx + dma + vtcm.
 * Tiles the FIR by output blocks of OT samples. For each tile: DMA the input
 * window (OT + ntaps-1 bytes, i.e. with halo) from DDR into VTCM, run the FIR on
 * the fast on-chip copy writing int32 results into a VTCM output slot, then DMA
 * the output slot back to DDR. The next input window is prefetched (async DMA)
 * while the current tile computes, hiding DDR latency behind compute. Both the
 * input reads and the int32 output writes hit VTCM instead of DDR. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define OT   4096                 /* outputs per tile */
#define HALO 7                    /* ntaps-1 */
#define ITILE (OT + HALO)         /* input bytes per tile */
#define ISLOT 8192                /* input slot stride (bytes) */
#define OSLOT (OT * 4)            /* output slot bytes = 16384 */

static desc_t d_in, d_pf, d_out;

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}
static inline void fir64(const int8_t *xp, const int8_t *taps, int ntaps,
                          HVX_Vector *lo, HVX_Vector *hi) {
    HVX_VectorPair acc = Q6_W_vcombine_VV(Q6_V_vzero(), Q6_V_vzero());
    for (int j = 0; j < ntaps; j++) {
        HVX_Vector xb = load_ua(xp + j);
        HVX_Vector xh = Q6_V_lo_W(Q6_Wh_vunpack_Vb(xb));
        HVX_Vector tv = Q6_Vh_vsplat_R((int32_t)(int16_t)taps[j]);
        acc = Q6_Ww_vmpyacc_WwVhVh(acc, xh, tv);
    }
    HVX_VectorPair o = Q6_W_vshuff_VVR(Q6_V_hi_W(acc), Q6_V_lo_W(acc), -4);
    *lo = Q6_V_lo_W(o); *hi = Q6_V_hi_W(o);
}

void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out, int n, int ntaps) {
    const uint32_t vin0 = VTCM_BASE,          vin1 = VTCM_BASE + ISLOT;
    const uint32_t vout0 = VTCM_BASE + 2*ISLOT, vout1 = VTCM_BASE + 2*ISLOT + OSLOT;

    int ntile = n / OT;
    int done  = ntile * OT;

    if (ntile == 0) {   /* tiny n: scalar fallback */
        for (int i = 0; i < n; i++) {
            int32_t acc = 0; for (int j = 0; j < ntaps; j++) acc += (int32_t)x[i+j]*(int32_t)taps[j];
            out[i] = acc;
        }
        return;
    }

    /* Prologue: bring input window for tile 0. */
    d_in.next = 0; d_in.ctrl = ITILE; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vin0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int t = 0; t < ntile; t++) {
        uint32_t vin  = (t & 1) ? vin1  : vin0;
        uint32_t vnxt = (t & 1) ? vin0  : vin1;
        uint32_t vout = (t & 1) ? vout1 : vout0;

        /* Prefetch next input window (async) while we compute the current tile. */
        if (t + 1 < ntile) {
            d_pf.next = 0; d_pf.ctrl = ITILE;
            d_pf.src = (uint32_t)(uintptr_t)(x + (t + 1) * OT); d_pf.dst = vnxt;
            Q6_dmstart_A(&d_pf);
        }

        const int8_t *ip = (const int8_t *)(uintptr_t)vin;
        int32_t *op = (int32_t *)(uintptr_t)vout;
        for (int i = 0; i < OT; i += 128) {
            HVX_Vector lo0, hi0, lo1, hi1;
            fir64(ip + i,      taps, ntaps, &lo0, &hi0);
            fir64(ip + i + 64, taps, ntaps, &lo1, &hi1);
            *(HVX_Vector *)(op + i)      = lo0;
            *(HVX_Vector *)(op + i + 32) = hi0;
            *(HVX_Vector *)(op + i + 64) = lo1;
            *(HVX_Vector *)(op + i + 96) = hi1;
        }

        if (t + 1 < ntile) Q6_R_dmwait();     /* finish prefetch before reuse */

        /* Stream this output tile back to DDR (blocking). */
        d_out.next = 0; d_out.ctrl = OSLOT;
        d_out.src = vout; d_out.dst = (uint32_t)(uintptr_t)(out + t * OT);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    /* Scalar tail for any leftover outputs (none when n % OT == 0). */
    for (int i = done; i < n; i++) {
        int32_t acc = 0; for (int j = 0; j < ntaps; j++) acc += (int32_t)x[i+j]*(int32_t)taps[j];
        out[i] = acc;
    }
}
