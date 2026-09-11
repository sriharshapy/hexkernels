/* sol_05: gather into 128B-chunk temp, HVX add per chunk â€” fused chunk idiom
 * Process in 128-element chunks: gather chunk â†’ HVX add with in_b chunk â†’ store.
 * Minimizes temp buffer size and interleaves gather and vector add.
 */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in_a, const int8_t *in_b,
                      const int32_t *idx, int8_t *out, int n) {
    const int vlen = 128;
    __attribute__((aligned(128))) int8_t tmp[128];

    int i = 0;
    for (; i + vlen <= n; i += vlen) {
        /* Gather 128 bytes scalar */
        for (int k = 0; k < vlen; k++)
            tmp[k] = in_a[idx[i + k]];
        /* HVX add */
        HVX_Vector va = *(const HVX_Vector *)tmp;
        HVX_Vector vb = *(const HVX_Vector *)(in_b + i);
        *(HVX_Vector *)(out + i) = Q6_Vb_vadd_VbVb(va, vb);
    }
    /* scalar tail */
    for (; i < n; i++)
        out[i] = (int8_t)((int16_t)in_a[idx[i]] + (int16_t)in_b[i]);
}