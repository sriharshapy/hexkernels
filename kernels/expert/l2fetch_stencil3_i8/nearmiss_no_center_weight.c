/* Near-miss: uses a plain (1,1,1) 3-tap average instead of the pinned
 * (1,2,1) weighted stencil -- forgets to double the center tap. Wrong for
 * almost every non-constant-neighborhood element. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

typedef long HEXAGON_Vect_UN __attribute__((__vector_size__(128))) __attribute__((aligned(4)));
#define vmemu(A) (*((const HEXAGON_Vect_UN *)(A)))

static inline HVX_Vector stencil3(HVX_Vector vprev, HVX_Vector vcur, HVX_Vector vnext) {
    HVX_VectorPair wp = Q6_Wh_vunpack_Vb(vprev);
    HVX_VectorPair wc = Q6_Wh_vunpack_Vb(vcur);
    HVX_VectorPair wn = Q6_Wh_vunpack_Vb(vnext);
    HVX_Vector plo = Q6_V_lo_W(wp), phi = Q6_V_hi_W(wp);
    HVX_Vector clo = Q6_V_lo_W(wc), chi = Q6_V_hi_W(wc);
    HVX_Vector nlo = Q6_V_lo_W(wn), nhi = Q6_V_hi_W(wn);
    HVX_Vector slo = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(plo, nlo), clo);   /* missing +clo again */
    HVX_Vector shi = Q6_Vh_vadd_VhVh(Q6_Vh_vadd_VhVh(phi, nhi), chi);
    return Q6_Vb_vpack_VhVh_sat(shi, slo);
}

void candidate_kernel(const int8_t *a, int8_t *out, int n) {
    if (n <= 0) return;
    { int32_t t = (int32_t)a[0] + (int32_t)a[0] + (int32_t)(n>1?a[1]:a[0]);
      if (t>127) t=127; if (t<-128) t=-128; out[0] = (int8_t)t; }

    int i = 1;
    for (; i + 128 <= n - 1; i += 128) {
        HVX_Vector vprev = (HVX_Vector)vmemu(a + i - 1);
        HVX_Vector vcur  = (HVX_Vector)vmemu(a + i);
        HVX_Vector vnext = (HVX_Vector)vmemu(a + i + 1);
        HEXAGON_Vect_UN r = (HEXAGON_Vect_UN)stencil3(vprev, vcur, vnext);
        __builtin_memcpy(out + i, &r, 128);
    }
    for (; i < n - 1; i++) {
        int32_t t = (int32_t)a[i-1] + (int32_t)a[i] + (int32_t)a[i+1];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[i] = (int8_t)t;
    }
    if (n > 1) {
        int j = n - 1;
        int32_t t = (int32_t)a[j-1] + (int32_t)a[j] + (int32_t)a[j];
        if (t > 127) t = 127; if (t < -128) t = -128;
        out[j] = (int8_t)t;
    }
}
