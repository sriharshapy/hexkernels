/* Solution 1 (= expert): hvx + rolling l2fetch + 4x unroll. Every 2KB of
 * progress, prefetch the next 2KB block of BOTH a[] and b[] a fixed
 * distance ahead -- rolled continuously for the whole loop, not just once.
 * l2fetch has no C intrinsic on this toolchain -> inline asm. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline void l2fetch_2k(const void *p) {
    uint64_t desc = ((uint64_t)16 << 32) | ((uint64_t)128 << 16) | (uint64_t)128;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(desc) : "memory");
}

void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    const int vlen = 128;
    const int PF = 16384;
    int i = 0;
    for (; i + 4*vlen <= n; i += 4*vlen) {
        if ((i & 2047) == 0 && i + PF + 2048 <= n) {
            l2fetch_2k(a + i + PF);
            l2fetch_2k(b + i + PF);
        }
        const HVX_Vector *pa = (const HVX_Vector*)(a + i);
        const HVX_Vector *pb = (const HVX_Vector*)(b + i);
        HVX_Vector *po = (HVX_Vector*)(out + i);
        po[0] = Q6_Vb_vadd_VbVb(pa[0], pb[0]);
        po[1] = Q6_Vb_vadd_VbVb(pa[1], pb[1]);
        po[2] = Q6_Vb_vadd_VbVb(pa[2], pb[2]);
        po[3] = Q6_Vb_vadd_VbVb(pa[3], pb[3]);
    }
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(*(const HVX_Vector *)(a + i), *(const HVX_Vector *)(b + i));
    for (; i < n; i++) out[i] = (int8_t)(a[i] + b[i]);
}
