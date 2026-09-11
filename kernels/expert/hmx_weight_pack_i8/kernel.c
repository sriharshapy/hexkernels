/* Expert: contraction-group 4-deep pack with inline arithmetic (same as s2).
 * Pure index remap -- 1.0x drill teaching the HMX int8 weight layout. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const int8_t *B, int8_t *out, int n) {
    for (int kg = 0; kg < n / 4; kg++) {
        int8_t *base = out + kg * 128;
        for (int km = 0; km < 4; km++) {
            const int8_t *brow = B + (kg*4 + km) * n;
            for (int j = 0; j < n; j++)
                base[j*4 + km] = brow[j];
        }
    }
}
