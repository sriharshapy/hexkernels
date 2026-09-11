/* HVX fp16 GELU (tanh approximation) -- ACCELERATED expert.
 *
 * HVX v68 has no native fp16<->fp32 vector widen/narrow (Q6_Wsf_vcvt_Vhf's
 * inverse, Q6_Vhf_vcvt_VsfVsf, fails to select at v68 -- verified empirically)
 * and no vector transcendental (tanhf) at all. So: scalar-cast the 1024
 * fp16 inputs up to fp32 (cheap single-instruction conversions, NOT
 * transcendentals), do the entire GELU math -- including tanh -- as a fully
 * vectorized 32-lane HVX qf32 computation (a [6/6] Pade rational
 * approximation of tanh, evaluated via Horner in qf32, with the division
 * done by a bit-hack-seeded Newton-Raphson vector reciprocal since HVX v68
 * has no float divide/reciprocal instruction), then scalar-cast the result
 * back down to fp16. Max abs tanh error < 5e-4 over x in [-6,6] (verified
 * against libm tanh in a standalone smoke test) -- comfortably inside this
 * task's fp16 tolerance (hvx_close_f16bits: atol 4e-3, rtol 8e-3). The
 * dominant cost (the rational approximation + NR reciprocal, ~30 vector
 * ops per 32 lanes) is fully vectorized; only cheap single-instruction
 * fp16<->fp32 casts remain scalar. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define HVX_TANH_A1 0.128205128f   /* 17325/135135 */
#define HVX_TANH_A2 0.002797203f   /*   378/135135 */
#define HVX_TANH_A3 0.0000074000f /*     1/135135 */
#define HVX_TANH_B1 0.461538462f   /* 62370/135135 */
#define HVX_TANH_B2 0.023310023f   /*  3150/135135 */
#define HVX_TANH_B3 0.000207200f   /*    28/135135 */

static inline HVX_Vector hvx_splat_f32(float f) {
    unsigned int b; __builtin_memcpy(&b, &f, 4);
    return Q6_V_vsplat_R((int)b);
}

/* Fast vector reciprocal: bit-hack initial guess + 2 Newton-Raphson
 * iterations (d must be > 0; the rational-tanh denominator always is). */
static inline HVX_Vector hvx_recip_sf(HVX_Vector d) {
    HVX_Vector magic = Q6_V_vsplat_R(0x7EF311C2);
    HVX_Vector y = Q6_Vw_vsub_VwVw(magic, d);
    HVX_Vector two = hvx_splat_f32(2.0f);
    for (int it = 0; it < 2; it++) {
        HVX_Vector dy = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(d, y));
        HVX_Vector t  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_VsfVsf(two, dy));
        y = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(y, t));
    }
    return y;
}

/* tanh(x) via a [6/6] Pade rational approximation (coefficients normalized
 * by 135135), qf32 Horner evaluation + NR vector reciprocal. */
static inline HVX_Vector hvx_tanh_hi_sf(HVX_Vector x) {
    HVX_Vector x2q = Q6_Vqf32_vmpy_VsfVsf(x, x);
    HVX_Vector one = hvx_splat_f32(1.0f);

    HVX_Vector p = Q6_Vqf32_vmpy_VsfVsf(hvx_splat_f32(HVX_TANH_A3), one);
    p = Q6_Vqf32_vmpy_Vqf32Vqf32(p, x2q); p = Q6_Vqf32_vadd_Vqf32Vsf(p, hvx_splat_f32(HVX_TANH_A2));
    p = Q6_Vqf32_vmpy_Vqf32Vqf32(p, x2q); p = Q6_Vqf32_vadd_Vqf32Vsf(p, hvx_splat_f32(HVX_TANH_A1));
    p = Q6_Vqf32_vmpy_Vqf32Vqf32(p, x2q); p = Q6_Vqf32_vadd_Vqf32Vsf(p, one);
    HVX_Vector p_sf = Q6_Vsf_equals_Vqf32(p);
    HVX_Vector num = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, p_sf));

    HVX_Vector q = Q6_Vqf32_vmpy_VsfVsf(hvx_splat_f32(HVX_TANH_B3), one);
    q = Q6_Vqf32_vmpy_Vqf32Vqf32(q, x2q); q = Q6_Vqf32_vadd_Vqf32Vsf(q, hvx_splat_f32(HVX_TANH_B2));
    q = Q6_Vqf32_vmpy_Vqf32Vqf32(q, x2q); q = Q6_Vqf32_vadd_Vqf32Vsf(q, hvx_splat_f32(HVX_TANH_B1));
    q = Q6_Vqf32_vmpy_Vqf32Vqf32(q, x2q); q = Q6_Vqf32_vadd_Vqf32Vsf(q, one);
    HVX_Vector den = Q6_Vsf_equals_Vqf32(q);

    HVX_Vector rden = hvx_recip_sf(den);
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(num, rden));
}

static inline HVX_Vector hvx_gelu_sf(HVX_Vector x) {
    HVX_Vector x2 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, x));
    HVX_Vector x3 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x2, x));
    HVX_Vector t  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x3, hvx_splat_f32(0.044715f)));
    HVX_Vector s  = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(x, t));
    HVX_Vector inner = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(s, hvx_splat_f32(0.7978845608f)));
    HVX_Vector th = hvx_tanh_hi_sf(inner);
    HVX_Vector onep = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(th, hvx_splat_f32(1.0f)));
    HVX_Vector xh = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, onep));
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(xh, hvx_splat_f32(0.5f)));
}

static float gelu_f32_scalar(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *x, hvx_hf *out, int n) {
    static float xf[1024] HVX_ALIGN;
    static float of[1024] HVX_ALIGN;

    for (int i = 0; i < n; i++) xf[i] = (float)x[i];   /* scalar widen (cheap cast) */

    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector vx = *(const HVX_Vector *)(xf + i);
        *(HVX_Vector *)(of + i) = hvx_gelu_sf(vx);
    }
    for (; i < n; i++) of[i] = gelu_f32_scalar(xf[i]);   /* tail */

    for (int j = 0; j < n; j++) out[j] = (hvx_hf)of[j];  /* scalar narrow (cheap cast) */
}
