/* EXPERT (achievability bar) — gather + vtcm (+ dma to stage the table).
 * Stage the whole table into VTCM once (uDMA), then use the HVX hardware gather
 * (Q6_vgather_ARMVw) to fetch 32 words per instruction from the VTCM-resident
 * table using runtime index offsets. VTCM random access is fast; the scalar
 * baseline pays DDR-miss latency on every scattered lookup. A dummy read of the
 * gather destination stalls until the gather completes (required sync). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
static desc_t dtab;

void candidate_kernel(const int32_t *table, const int32_t *idx, int32_t *out,
                      int table_words, int n_idx) {
    uint32_t tab = VTCM_BASE;                 /* table region in VTCM */
    uint32_t dst = VTCM_BASE + (uint32_t)table_words * 4u; /* 32-word gather dest */
    uint32_t region = (uint32_t)table_words * 4u - 1u;     /* byte-offset mask */

    dtab.next = 0; dtab.ctrl = (uint32_t)table_words * 4u;
    dtab.src = (uint32_t)(uintptr_t)table; dtab.dst = tab;
    Q6_dmstart_A(&dtab); Q6_R_dmwait();

    HVX_Vector *vd = (HVX_Vector*)(uintptr_t)dst;
    int i = 0;
    for (; i + 32 <= n_idx; i += 32) {
        HVX_Vector vi   = *(const HVX_Vector*)(idx + i);   /* 32 indices */
        HVX_Vector voff = Q6_Vw_vasl_VwR(vi, 2);           /* *4 -> byte offsets */
        Q6_vgather_ARMVw(vd, tab, region, voff);
        volatile HVX_Vector sync = *(volatile HVX_Vector*)vd; (void)sync;
        *(HVX_Vector*)(out + i) = *vd;
    }
    /* scalar tail (n_idx not a multiple of 32) */
    for (; i < n_idx; i++) out[i] = ((const int32_t*)(uintptr_t)tab)[idx[i]];
}
