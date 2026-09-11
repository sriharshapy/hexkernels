/* HMX fp16 32x32 matmul + GELU epilogue (tolerance-correct) -- the
 * ACCELERATED expert. Packs A,B into the fp16 crouton layout
 * (off(r,c)=(r/2)*64+c*2+(r&1)), runs one HMX matmul, reads the crouton result
 * back, then applies the (necessarily scalar -- no HVX transcendental) GELU
 * per element. Scalar VTCM accesses cost ~48 cycles each in timing mode, so
 * pack/unpack is STAGED through cacheable buffers and moved to/from VTCM in
 * bulk 128B vector copies. Recipe: m1_driver/capability/probes/hmx_fp16_matmul.c. */
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
    hvx_hf *vB = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x0800u);  /* 2048B apart */
    hvx_hf *vO = (hvx_hf *)(uintptr_t)(HVX_VTCM_BASE + 0x1000u);

    static hvx_hf aA[32*32] HVX_ALIGN, aB[32*32] HVX_ALIGN, aO[32*32] HVX_ALIGN;

    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            aA[hvx_crouton_off(r, c)] = A[r*n + c];
            aB[hvx_crouton_off(r, c)] = B[r*n + c];
        }
    { HVX_Vector *s = (HVX_Vector *)aA, *d = (HVX_Vector *)vA;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    { HVX_Vector *s = (HVX_Vector *)aB, *d = (HVX_Vector *)vB;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }

    Q6_mxclracc_hf();
    Q6_activation_hf_mxmem_RR((unsigned int)(uintptr_t)vA, 2047);
    Q6_weight_hf_mxmem_RR((unsigned int)(uintptr_t)vB, 2047);
    Q6_mxmem_AR_after_hf(vO, 2047);
    __asm__ volatile("isync\n\t");

    { HVX_Vector *s = (HVX_Vector *)vO, *d = (HVX_Vector *)aO;
      for (int b = 0; b < 16; b++) d[b] = s[b]; }
    for (int r = 0; r < n; r++)
        for (int c = 0; c < n; c++) {
            hvx_hf m = aO[hvx_crouton_off(r, c)];
            out[r*n + c] = (hvx_hf)gelu_f32((float)m);
        }
}
