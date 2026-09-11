/* HMX FP16 32x32 matmul + GELU epilogue.
 * GELU = 0.5*x*(1+tanh(0.7978845608*(x+0.044715*x^3))), applied in float
 * to the fp16-rounded HMX output, then cast back to fp16.
 * Same constants as harness reference -> bit-exact. */
#include "kernel_api.h"
#include "harness_common.h"
#include <hexagon_types.h>
#include <hmx_hexagon_protos.h>
#include <math.h>

static float gelu_f32(float x) {
    float c = 0.7978845608f * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + tanhf(c));
}

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n) {
    hvx_hf *vA = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0000u);
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    /* Pack croutons */
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            vA[hvx_crouton_off(r, c)] = A[r*n+c];
            vB[hvx_crouton_off(r, c)] = B[r*n+c];
        }

    /* HMX matmul */
    Q6_mxclracc_hf();
    Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    /* Unpack crouton + apply GELU in float, round back to fp16 */
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            hvx_hf m = vO[hvx_crouton_off(r, c)];   /* fp16-rounded HMX result */
            out[r*n+c] = (hvx_hf)gelu_f32((float)m);
        }
}
