/* EXPERT (achievability bar) -- dma + vtcm (+ hvx).
 * Stage the WHOLE table into VTCM once (a single uDMA transfer), then for
 * each row do a fast 128-byte HVX vector load from the VTCM-resident table
 * row and a vector store to the output row. VTCM random-row access is fast;
 * the scalar baseline pays a DDR-miss per lookup. E is fixed to a multiple
 * of 128 (one HVX vector) so each row copy is exactly one aligned vector
 * op. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef struct __attribute__((aligned(32))) { uint32_t next, ctrl, src, dst; } desc_t;
#define VTCM_BASE 0xd8400000u
static desc_t dtab;

void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E, int vocab_size) {
    uint32_t tab_base = VTCM_BASE;
    uint32_t tab_bytes = (uint32_t)vocab_size * (uint32_t)E;

    dtab.next = 0; dtab.ctrl = tab_bytes;
    dtab.src = (uint32_t)(uintptr_t)table; dtab.dst = tab_base;
    Q6_dmstart_A(&dtab); Q6_R_dmwait();

    const int8_t *vtab = (const int8_t *)(uintptr_t)tab_base;
    const int vlen = sizeof(HVX_Vector);   /* == 128 == E */

    for (int i = 0; i < T; i++) {
        const int8_t *src = vtab + (int64_t)idx[i] * E;
        int8_t       *dst = out  + (int64_t)i * E;
        int e = 0;
        for (; e + vlen <= E; e += vlen)
            *(HVX_Vector *)(dst + e) = *(const HVX_Vector *)(src + e);
        for (; e < E; e++) dst[e] = src[e];
    }
}
