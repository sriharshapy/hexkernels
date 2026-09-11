/* EXPERT (achievability bar) — gather + vtcm (+ dma to stage the table).
 * Stage the int16 table in VTCM (uDMA), then use the HVX halfword hardware
 * gather (Q6_vgather_ARMVh) to fetch 64 halfwords per instruction from the
 * VTCM-resident table. Offsets are halfword byte-offsets (idx*2). A dummy read
 * of the gather dest stalls until completion. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
static desc_t dtab;

void candidate_kernel(const int16_t *table, const int16_t *idx, int16_t *out,
                      int table_hwords, int n_idx) {
    uint32_t tab = VTCM_BASE;
    uint32_t dst = VTCM_BASE + (uint32_t)table_hwords * 2u;   /* 64-hword gather dest */
    uint32_t region = (uint32_t)table_hwords * 2u - 1u;       /* byte-offset mask */

    dtab.next = 0; dtab.ctrl = (uint32_t)table_hwords * 2u;
    dtab.src = (uint32_t)(uintptr_t)table; dtab.dst = tab;
    Q6_dmstart_A(&dtab); Q6_R_dmwait();

    HVX_Vector *vd = (HVX_Vector*)(uintptr_t)dst;
    int i = 0;
    for (; i + 64 <= n_idx; i += 64) {
        HVX_Vector vi   = *(const HVX_Vector*)(idx + i);   /* 64 int16 indices */
        HVX_Vector voff = Q6_Vh_vasl_VhR(vi, 1);           /* *2 -> halfword byte offsets */
        Q6_vgather_ARMVh(vd, tab, region, voff);
        volatile HVX_Vector sync = *(volatile HVX_Vector*)vd; (void)sync;
        *(HVX_Vector*)(out + i) = *vd;                     /* 64 halfwords to DDR */
    }
    for (; i < n_idx; i++) out[i] = ((const int16_t*)(uintptr_t)tab)[idx[i]];
}
