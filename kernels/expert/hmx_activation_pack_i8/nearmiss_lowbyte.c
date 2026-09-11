/* Near-miss: packs the activation into the LOW byte of each fp16 crouton slot
 * (offset 2*crouton_off, no +1) instead of the HIGH byte. Compiles and touches
 * the right slots, but the int8 lands in the dead byte -> the HIGH bytes stay 0
 * and every non-zero value is misplaced -> must FAIL bit-exact. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, uint8_t *out, int n) {
    for (int i = 0; i < 2048; i++) out[i] = 0;
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++)
            out[2 * hvx_crouton_off(i, k)] = A[i*n + k];   /* LOW byte -- wrong */
}
