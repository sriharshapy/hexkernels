/* EXPERT (achievability bar) -- hvx + dma + vtcm.
 * DMA-double-buffers each row's INPUT through VTCM: prefetch row r+1's input
 * while row r's data is bulk vector-copied out of VTCM into cacheable
 * scratch, all three on-chip passes (max-scan, sum+cache-e_j, normalize)
 * run against that cacheable copy, the result is bulk vector-copied into a
 * VTCM output slot, and DMA'd back to DDR. Turns baseline.c's ~4 DDR
 * passes/row into 2 bulk DMA transfers/row (1 in, 1 out).
 *
 * IMPORTANT: a SCALAR access to a VTCM address costs a documented ~48
 * cycles/access fixed penalty in this timing model (far worse than a scalar
 * DDR/regular-memory access) -- so the three scalar compute passes must
 * NEVER touch the VTCM address directly. Each row's bytes are bulk
 * VECTOR-copied (128B at a time -- vector loads/stores to VTCM are cheap)
 * into/out of small cacheable scratch buffers, and all scalar/LUT work runs
 * against those (regular, cache-friendly memory). Same "pack/unpack in
 * cacheable buffers, move to/from VTCM in bulk vector copies" pattern used
 * for HMX crouton pack/unpack and vtcm_stage_matmul_i8. */
#include <stdint.h>
#include <stddef.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;

#define VTCM_BASE 0xd8400000u
#define ROWBUF 131072   /* >= max C*2 bytes (65500*2=131000), one row-buffer slot */
#define CACHE_ELEMS (ROWBUF / 2)   /* int16/uint16 elements per cacheable buffer */

static desc_t d_in, d_pf, d_out;

/* Cacheable scratch: all scalar compute hits THESE, never VTCM directly. */
static int16_t  __attribute__((aligned(128))) x_cache[CACHE_ELEMS];
static uint16_t __attribute__((aligned(128))) e_cache[CACHE_ELEMS];
static int16_t  __attribute__((aligned(128))) o_cache[CACHE_ELEMS];

static void softmax_row_cache(const int16_t *xr, uint16_t *er, int16_t *outr,
                              int C, const uint16_t *lut) {
    int16_t m = xr[0];
    for (int j = 1; j < C; j++) if (xr[j] > m) m = xr[j];

    int64_t S = 0;
    for (int j = 0; j < C; j++) {
        int32_t diff = (int32_t)xr[j] - (int32_t)m;
        if (diff < -255) diff = -255;
        uint16_t e = lut[diff + 255];
        er[j] = e;
        S += (int64_t)e;
    }

    int64_t halfS = S / 2;
    for (int j = 0; j < C; j++) {
        int64_t num = (int64_t)er[j] * 32767 + halfS;
        outr[j] = (int16_t)(num / S);
    }
}

void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const uint16_t *exp_lut) {
    const uint32_t CB = (uint32_t)C * 2u;
    const uint32_t vt_x0 = VTCM_BASE,            vt_x1 = VTCM_BASE + ROWBUF;
    const uint32_t vt_o0 = VTCM_BASE + 2*ROWBUF, vt_o1 = VTCM_BASE + 3*ROWBUF;
    const int nvec = ROWBUF / 128;   /* bulk vector-copy granularity */

    if (R == 0) return;

    d_in.next = 0; d_in.ctrl = CB; d_in.src = (uint32_t)(uintptr_t)x; d_in.dst = vt_x0;
    Q6_dmstart_A(&d_in); Q6_R_dmwait();

    for (int r = 0; r < R; r++) {
        uint32_t cur_x = (r & 1) ? vt_x1 : vt_x0;
        uint32_t cur_o = (r & 1) ? vt_o1 : vt_o0;
        uint32_t nxt_x = (r & 1) ? vt_x0 : vt_x1;

        if (r + 1 < R) {
            d_pf.next = 0; d_pf.ctrl = CB;
            d_pf.src = (uint32_t)(uintptr_t)(x + (size_t)(r + 1) * C);
            d_pf.dst = nxt_x;
            Q6_dmstart_A(&d_pf);
        }

        /* Bulk vector-copy this row's input out of VTCM into cacheable x_cache. */
        {
            const HVX_Vector *vsrc = (const HVX_Vector *)(uintptr_t)cur_x;
            HVX_Vector *vdst = (HVX_Vector *)x_cache;
            for (int v = 0; v < nvec; v++) vdst[v] = vsrc[v];
        }

        softmax_row_cache(x_cache, e_cache, o_cache, C, exp_lut);

        /* Bulk vector-copy the cacheable result into the VTCM output slot. */
        {
            const HVX_Vector *vsrc = (const HVX_Vector *)o_cache;
            HVX_Vector *vdst = (HVX_Vector *)(uintptr_t)cur_o;
            for (int v = 0; v < nvec; v++) vdst[v] = vsrc[v];
        }

        if (r + 1 < R) Q6_R_dmwait();

        d_out.next = 0; d_out.ctrl = CB;
        d_out.src = cur_o; d_out.dst = (uint32_t)(uintptr_t)(out + (size_t)r * C);
        Q6_dmstart_A(&d_out); Q6_R_dmwait();
    }
}
