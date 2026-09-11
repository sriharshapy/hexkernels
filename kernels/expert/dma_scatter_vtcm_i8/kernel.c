/* Solution 1 (= expert): scatter + vtcm (+ dma to drain the table). Scatter
 * 32 widened (int8->int32) values per instruction directly into a
 * VTCM-resident table using the HVX hardware scatter (Q6_vscatter_RMVwV),
 * then DMA the completed table out to DDR in one bulk transfer -- avoiding
 * n individually-scattered DDR stores. A dummy (volatile) read of the
 * table after the last scatter forces the writes to complete before the
 * DMA-out reads them (mirrors the gather completion-sync idiom). */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
static desc_t dtab;

void candidate_kernel(const int8_t *values, const int32_t *idx, int32_t *out, int n) {
    uint32_t tab = VTCM_BASE;
    uint32_t region = (uint32_t)n * 4u - 1u;

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector vidx = *(const HVX_Vector *)(idx + i);
        HVX_Vector voff = Q6_Vw_vasl_VwR(vidx, 2);   /* index*4 -> byte offset */

        int32_t wide[32] __attribute__((aligned(128)));
        for (int j = 0; j < 32; j++) wide[j] = (int32_t)values[i + j];
        HVX_Vector vdata = *(HVX_Vector *)wide;

        Q6_vscatter_RMVwV(tab, region, voff, vdata);
    }
    /* Force the scattered writes to complete before the DMA-out reads them. */
    volatile int32_t sync = *(volatile int32_t *)(uintptr_t)tab; (void)sync;

    for (; i < n; i++) *(int32_t *)(uintptr_t)(tab + (uint32_t)idx[i] * 4u) = (int32_t)values[i];

    dtab.next = 0; dtab.ctrl = (uint32_t)n * 4u;
    dtab.src = tab; dtab.dst = (uint32_t)(uintptr_t)out;
    Q6_dmstart_A(&dtab); Q6_R_dmwait();
}
