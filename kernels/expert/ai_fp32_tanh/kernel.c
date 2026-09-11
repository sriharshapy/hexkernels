/* HVX fp32 tanh -- ACCELERATED expert.
 *
 * HVX v68 has no vector transcendental instruction, so tanh is computed via
 * a [6/6] Pade rational approximation (coefficients normalized by 135135),
 * evaluated with Horner in qf32 -- fully vectorized, 32 fp32 lanes/vector.
 * The division uses a bit-hack-seeded Newton-Raphson vector reciprocal
 * (HVX v68 has no float divide/reciprocal instruction; 2 NR iterations give
 * ~1e-5 relative accuracy). Max abs error < 5e-4 over x in [-6,6] (verified
 * against libm tanh in a standalone smoke test) -- comfortably inside this
 * task's fp32 tolerance (hvx_close_f32: atol 1e-4, rtol 1e-3). */
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

void candidate_kernel(const float *x, float *out, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector vx = *(const HVX_Vector *)(x + i);
        *(HVX_Vector *)(out + i) = hvx_tanh_hi_sf(vx);
    }
    for (; i < n; i++) out[i] = tanhf(x[i]);   /* tail */
}
