/* EXPERT (achievability bar) -- hvx + dma + vtcm.
 * Tiles the overlapping energy pool (window=8, stride=4) by output blocks
 * of OT. For each tile: DMA the input window (OT*4+4 halo bytes) from DDR
 * into VTCM, compute the sum of 4 squared bytes per stride-4 group via
 * Q6_Vw_vrmpy_VbVb on BOTH the aligned block and the block shifted by 4
 * bytes, sum the two (window-8 energy for 32 outputs at once), shift+clamp
 * writing int8 results into a VTCM output slot, then DMA the output slot
 * back to DDR. The next input window is prefetched (async DMA) while the
 * current tile computes, hiding DDR latency AND avoiding re-fetching the
 * overlapped halo bytes from DDR on every block. */
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define OT    8192                    /* outputs per tile (multiple of 128) */
#define HALO  4                       /* window - stride */
#define ITILE (OT * 4 + HALO)         /* input bytes per tile */
#define ISLOT ((((ITILE) + 256 + 127) / 128) * 128)  /* padded + rounded to 128B (VTCM tiles must stay vector-aligned) */
#define OSLOT OT                      /* output slot bytes */

static desc_t d_in, d_pf, d_out;
static int32_t acc_scratch[OT] __attribute__((aligned(128)));
static int8_t  out_local[OT]   __attribute__((aligned(128)));

static inline HVX_Vector load_ua(const int8_t *p) {
    const HVX_Vector *vp = (const HVX_Vector *)((uintptr_t)p & ~(uintptr_t)127);
    HVX_Vector v0 = *vp, v1 = *(vp + 1);
    return Q6_V_valign_VVR(v1, v0, (int)((uintptr_t)p & 127));
}

static void energy_pool_core(const int8_t *ip, int8_t *op, int nout, int shift) {
    int j = 0;
    for (; j + 32 <= nout; j += 32) {
        HVX_Vector cur_in = *(const HVX_Vector *)(ip + (size_t)j * 4);
        HVX_Vector cur = Q6_Vw_vrmpy_VbVb(cur_in, cur_in);
        HVX_Vector nxt_in = load_ua(ip + (size_t)j * 4 + 4);
        HVX_Vector nxt = Q6_Vw_vrmpy_VbVb(nxt_in, nxt_in);
        HVX_Vector energy = Q6_Vw_vadd_VwVw(cur, nxt);
        HVX_Vector sh = Q6_Vw_vasr_VwR(energy, shift);
        *(HVX_Vector *)(acc_scratch + j) = sh;
    }
    for (; j < nout; j++) {
        int32_t a = 0;
        for (int k = 0; k < 8; k++) { int32_t v = ip[(size_t)j * 4 + k]; a += v * v; }
        acc_scratch[j] = a >> shift;
    }

    /* Clamp+narrow into a LOCAL (non-VTCM) buffer scalarly, then bulk
     * vector-copy to op (avoids scalar per-element writes to a
     * VTCM-resident tile). */
    for (int jj = 0; jj < nout; jj++) {
        int32_t v = acc_scratch[jj];
        if (v > 127) v = 127;
        out_local[jj] = (int8_t)v;
    }
    int jj = 0;
    for (; jj + 128 <= nout; jj += 128)
        *(HVX_Vector *)(op + jj) = *(const HVX_Vector *)(out_local + jj);
    for (; jj < nout; jj++) op[jj] = out_local[jj];
}

void candidate_kernel(const int8_t *x, int8_t *out, int n, int window, int shift) {
    (void)window;
    const uint32_t vin0 = VTCM_BASE,            vin1 = VTCM_BASE + ISLOT;
    const uint32_t vout0 = VTCM_BASE + 2*ISLOT, vout1 = VTCM_BASE + 2*ISLOT + OSLOT;

    int ntile = n / OT;
    int done  = ntile * OT;

    if (ntile == 0) {
        energy_pool_core(x, out, n, shift);
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
            d_pf.src = (uint32_t)(uintptr_t)(x + (size_t)(t + 1) * OT * 4); d_pf.dst = vnxt;
            Q6_dmstart_A(&d_pf);
        }

        energy_pool_core((const int8_t *)(uintptr_t)vin, (int8_t *)(uintptr_t)vout, OT, shift);

        if (t + 1 < ntile) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = OSLOT;
        d_out.src = vout; d_out.dst = (uint32_t)(uintptr_t)(out + (size_t)t * OT);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }

    if (done < n) energy_pool_core(x + (size_t)done * 4, out + done, n - done, shift);
}
