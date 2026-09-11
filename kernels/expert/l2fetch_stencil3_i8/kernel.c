/* Solution 1 (= expert): hvx 3-tap (1,2,1) edge-replicated stencil + rolling
 * l2fetch. Every 2KB of progress, prefetch the next 2KB block of a[] a fixed
 * distance ahead -- rolled continuously for the whole loop. l2fetch has no C
 * intrinsic on this toolchain -> inline asm. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) (*((const HEXAGON_Vect_UN *)(A)))

static inline void l2fetch_2k(const void *p) {
    uint64_t desc = ((uint64_t)16 << 32) | ((uint64_t)128 << 16) | (uint64_t)128;
    __asm__ __volatile__("l2fetch(%0,%1)" : : "r"(p), "r"(desc) : "memory");
}

static inline HVX_Vector stencil3(HVX_Vector vprev, HVX_Vector vcur, HVX_Vector vnext) {
    HVX_VectorPair wp = Q6_Wh_vunpack_Vb(vprev);
    HVX_VectorPair wc = Q6_Wh_vunpack_Vb(vcur);
    HVX_VectorPair wn = Q6_Wh_vunpack_Vb(vnext);
    HVX_Vector plo = Q6_V_lo_W(wp), phi = Q6_V_hi_W(wp);
    HVX_Vector clo = Q6_V_lo_W(wc), chi = Q6_V_hi_W(wc);
    HVX_Vector nlo = Q6_V_lo_W(wn), nhi = Q6_V_hi_W(wn);
    HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(plo, nlo), Q6_Vh_vadd_VhVh(clo, clo));
    HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(phi, nhi), Q6_Vh_vadd_VhVh(chi, chi));
    return Q6_Vb_vpack_VhVh_sat(shi, slo);
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    const int PF = 16384;
    if (n <= 0) return;
    { int32_t t = (int32_t)a[0] + 2*(int32_t)a[0] + (int32_t)(n>1?a[1]:a[0]);
      if (t>127) t=127; if (t<-128) t=-128; out[0] = (int8_t)t; }

    int i = 1;
    for (; i + 128 <= n - 1; i += 128) {
        if (((i - 1) & 2047) == 0 && i - 1 + PF + 2048 <= n) l2fetch_2k(a + i - 1 + PF);
        HVX_Vector vprev = (HVX_Vector)vmemu(a + i - 1);
        HVX_Vector vcur  = (HVX_Vector)vmemu(a + i);
        HVX_Vector vnext = (HVX_Vector)vmemu(a + i + 1);
        HEXAGON_Vect_UN r = (HEXAGON_Vect_UN)stencil3(vprev, vcur, vnext);
        __builtin_memcpy(out + i, &r, 128);
    }
    for (; i < n - 1; i++) {
        int32_t t = (int32_t)a[i-1] + 2*(int32_t)a[i] + (int32_t)a[i+1];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
    if (n > 1) {
        int j = n - 1;
        int32_t t = (int32_t)a[j-1] + 2*(int32_t)a[j] + (int32_t)a[j];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[j] = (int8_t)t;
    }
}
