/* Solution 1 (= expert): hvx scale-by-1.5-with-round + rolling l2fetch + 4x
 * unroll. Every 2KB of progress, prefetch the next 2KB block of a[] a fixed
 * distance ahead -- rolled continuously for the whole loop. l2fetch has no C
 * intrinsic on this toolchain -> inline asm. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline void l2fetch_2k(const void *p) {
    uint64_t desc = ((uint64_t)16 << 32) | ((uint64_t)128 << 16) | (uint64_t)128;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(desc) : "memory");
}

static inline HVX_Vector scale15(HVX_Vector va, HVX_Vector three, HVX_Vector one) {
    HVX_VectorPair w = Q6_Wh_vunpack_Vb(va);
    HVX_Vector lo = Q6_V_lo_W(w), hi = Q6_V_hi_W(w);
    lo = Q6_Vh_vmpyi_VhVh(lo, three);
    hi = Q6_Vh_vmpyi_VhVh(hi, three);
    lo = Q6_Vh_vadd_VhVh(lo, one);
    hi = Q6_Vh_vadd_VhVh(hi, one);
    lo = Q6_Vh_vasr_VhR(lo, 1);
    hi = Q6_Vh_vasr_VhR(hi, 1);
    return Q6_Vb_vpack_VhVh_sat(hi, lo);
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const int vlen = 128;
    const int PF = 16384;
    HVX_Vector three = Q6_Vh_vsplat_R(3);
    HVX_Vector one = Q6_Vh_vsplat_R(1);
    int i = 0;
    for (; i + 4*vlen <= n; i += 4*vlen) {
        if ((i & 2047) == 0 && i + PF + 2048 <= n) l2fetch_2k(a + i + PF);
        const HVX_Vector *pa = (const HVX_Vector*)(a + i);
        HVX_Vector *po = (HVX_Vector*)(out + i);
        po[0] = scale15(pa[0], three, one);
        po[1] = scale15(pa[1], three, one);
        po[2] = scale15(pa[2], three, one);
        po[3] = scale15(pa[3], three, one);
    }
    for (; i + vlen <= n; i += vlen)
        *(HVX_Vector *)(out + i) = scale15(*(const HVX_Vector *)(a + i), three, one);
    for (; i < n; i++) {
        int32_t t = (int32_t)a[i] * 3;
        t = (t + 1) >> 1;
        if (t > 127) t = 127;
        if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
}
