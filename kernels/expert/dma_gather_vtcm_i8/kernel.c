/* Solution 1 (= expert): gather + vtcm (+ dma to stage the table), narrow
 * output. Stage the whole table into VTCM once (uDMA), then use the HVX
 * hardware gather (Q6_vgather_ARMVw) to fetch 32 words per instruction from
 * the VTCM-resident table using runtime index offsets. A dummy read of the
 * gather destination stalls until the gather completes (required sync);
 * the 32 gathered int32 lanes are then narrowed to int8 with a short local
 * (register-array) loop -- cheap compared to the DDR-miss lookups the
 * scalar baseline pays per element. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
static desc_t dtab;

void candidate_kernel(const int32_t *table, const int32_t *idx, int8_t *out,
                      int table_words, int n_idx) {
    uint32_t tab = VTCM_BASE;
    uint32_t dst = VTCM_BASE + (uint32_t)table_words * 4u;
    uint32_t region = (uint32_t)table_words * 4u - 1u;

    dtab.next = 0; dtab.ctrl = (uint32_t)table_words * 4u;
    dtab.src = (uint32_t)(uintptr_t)table; dtab.dst = tab;
    Q6_dmstart_A(&dtab); Q6_R_dmwait();

    HVX_Vector *vd = (HVX_Vector*)(uintptr_t)dst;
    int i = 0;
    for (; i + 32 <= n_idx; i += 32) {
        HVX_Vector vi   = *(const HVX_Vector*)(idx + i);
        HVX_Vector voff = Q6_Vw_vasl_VwR(vi, 2);
        Q6_vgather_ARMVw(vd, tab, region, voff);
        volatile HVX_Vector sync = *(volatile HVX_Vector*)vd; (void)sync;

        int32_t lanes[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lanes = *vd;
        for (int j = 0; j < 32; j++) out[i + j] = (int8_t)lanes[j];
    }
    for (; i < n_idx; i++) out[i] = (int8_t)((const int32_t *)(uintptr_t)tab)[idx[i]];
}
