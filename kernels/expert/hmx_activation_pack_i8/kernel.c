/* Expert: row-pair pack with inline offset arithmetic (same as s2). A pure
 * index remap -- 1.0x drill, teaching the HMX int8 activation crouton layout. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *A, uint8_t *out, int n) {
    for (int i = 0; i < 2048; i++) out[i] = 0;
    for (int rp = 0; rp < n / 2; rp++) {
        const uint8_t *r0 = A + (2*rp + 0) * n;
        const uint8_t *r1 = A + (2*rp + 1) * n;
        uint8_t *base = out + rp * 128;
        for (int k = 0; k < n; k++) {
            base[k*4 + 1] = r0[k];
            base[k*4 + 3] = r1[k];
        }
    }
}
