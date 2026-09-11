/* EXPERT (achievability bar) — HVX row-gather.
 * Each embedding row is E bytes (E is a multiple of 128 = one HVX vector).
 * Replace the scalar per-byte copy with aligned 128-byte vector loads/stores:
 * out[i, :] = table[idx[i], :]. table/out rows are 128-byte aligned. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E) {
    const int vlen = sizeof(HVX_Vector); /* 128 */
    for (int i = 0; i < T; i++) {
        const int8_t *src = table + (int)idx[i] * E;
        int8_t       *dst = out   + i * E;
        int e = 0;
        for (; e + vlen <= E; e += vlen)
            *(HVX_Vector *)(dst + e) = *(const HVX_Vector *)(src + e);
        for (; e < E; e++) dst[e] = src[e];
    }
}
