/* EXPERT (achievability bar) — hvx + l2fetch + 4x unroll.
 * ReLU is a single sequential input stream, which the hardware L2 stream
 * prefetcher already handles well: l2fetch ALONE only buys ~1.12x here. The
 * decisive win on this bandwidth-bound pointwise op is 4x loop unrolling (more
 * in-flight loads / better ILP), with a rolling l2fetch layered on top.
 * Measured (N=524288): baseline 99849 -> unroll-only 65060 (1.53x) ->
 * unroll+l2fetch 62165 (1.61x kernel cycles).
 *
 * l2fetch has no C intrinsic on this toolchain -> inline asm. Register-pair
 * descriptor: [15:0]=stride [31:16]=width [47:32]=height (bytes/rows);
 * width=128,height=16,stride=128 -> one contiguous 2 KB block into L2.
 */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline void l2fetch_2k(const void *p) {
    uint64_t desc = ((uint64_t)16 << 32) | ((uint64_t)128 << 16) | (uint64_t)128;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(desc) : "memory");
}

void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    const int vlen = 128;
    const int PF = 16384;      /* prefetch distance (bytes) */
    HVX_Vector zero = Q6_V_vzero();
    int i = 0;
    for (; i + 4*vlen <= n; i += 4*vlen) {
        if ((i & 2047) == 0) if (i + PF + 2048 <= n) l2fetch_2k(x + i + PF);
        const HVX_Vector *px = (const HVX_Vector*)(x + i);
        HVX_Vector *po = (HVX_Vector*)(out + i);
        po[0] = Q6_Vb_vmax_VbVb(px[0], zero);
        po[1] = Q6_Vb_vmax_VbVb(px[1], zero);
        po[2] = Q6_Vb_vmax_VbVb(px[2], zero);
        po[3] = Q6_Vb_vmax_VbVb(px[3], zero);
    }
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = Q6_Vb_vmax_VbVb(*(const HVX_Vector *)(x + i), zero);
    for (; i < n; i++) out[i] = (x[i] > 0) ? x[i] : 0;
}
