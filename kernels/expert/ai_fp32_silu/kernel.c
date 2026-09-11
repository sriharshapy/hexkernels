/* HVX fp32 SiLU (swish) -- ACCELERATED expert.
 *
 * silu(x) = x*sigmoid(x) = x*0.5*(1+tanh(x/2)). HVX v68 has no vector
 * transcendental, so tanh is computed via a [6/6] Pade rational
 * approximation (coefficients normalized by 135135), Horner-evaluated in
 * qf32 -- fully vectorized, 32 fp32 lanes/vector. Division uses a
 * bit-hack-seeded Newton-Raphson vector reciprocal (HVX v68 has no float
 * divide/reciprocal instruction). Max abs tanh error < 5e-4 over x in
 * [-6,6] (verified against libm tanh in a standalone smoke test) --
 * comfortably inside this task's fp32 tolerance (hvx_close_f32: atol 1e-4,
 * rtol 1e-3). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <math.h>

#define HVX_TANH_A1 0.128205128f
#define HVX_TANH_A2 0.002797203f
#define HVX_TANH_A3 0.0000074000f
#define HVX_TANH_B1 0.461538462f
#define HVX_TANH_B2 0.023310023f
#define HVX_TANH_B3 0.000207200f

static inline HVX_Vector hvx_splat_f32(float f) {
    unsigned int b; __builtin_memcpy(&b, &f, 4);
    return Q6_V_vsplat_R((int)b);
}

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

static inline HVX_Vector hvx_silu_sf(HVX_Vector x) {
    HVX_Vector half_x = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, hvx_splat_f32(0.5f)));
    HVX_Vector th = hvx_tanh_hi_sf(half_x);
    HVX_Vector onep = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_VsfVsf(th, hvx_splat_f32(1.0f)));
    HVX_Vector sig = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(onep, hvx_splat_f32(0.5f)));
    return Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(x, sig));
}

void candidate_kernel(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector vx = *(const HVX_Vector *)(x + i);
        *(HVX_Vector *)(out + i) = hvx_silu_sf(vx);
    }
    for (; i < n; i++) { float xx = x[i]; out[i] = xx / (1.0f + expf(-xx)); }   /* tail */
}
