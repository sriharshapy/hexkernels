/* HVX fp32 elementwise (Hadamard) product -- ACCELERATED expert.
 *
 * Unlike the fp16 hadamard task (which has a bit-exact widening-multiply
 * path: Q6_Wqf32_vmpy_VhfVhf + Q6_Vhf_equals_Wqf32), HVX v68 has no
 * correctly-rounded native sf*sf vector multiply (Q6_Vsf_vmpy_VsfVsf fails
 * to select at v68 -- verified empirically) and the qf32 round-trip
 * (Q6_Vqf32_vmpy_VsfVsf + Q6_Vsf_equals_Vqf32) is NOT bit-exact for sf*sf
 * (qf32 doesn't carry enough extra precision beyond plain fp32 to re-round
 * losslessly -- measured ~1e-7 relative error per multiply). This is why
 * the harness was switched from bit-exact to hvx_close_f32 tolerance (atol
 * 1e-4, rtol 1e-3); the qf32 round-trip's ~1e-7 relative error is far
 * inside that tolerance. 32 fp32 lanes/vector; n=1000 needs a scalar tail
 * (1000 = 31*32 + 8). */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const float *a, const float *b, float *out, int n) {
    int i = 0;
    for (; i + 32 <= n; i += 32) {
        HVX_Vector va = *(const HVX_Vector *)(a + i);
        HVX_Vector vb = *(const HVX_Vector *)(b + i);
        *(HVX_Vector *)(out + i) = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] * b[i];   /* tail */
}
