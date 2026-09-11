/* NEAR-MISS: uses a plain truncating arithmetic shift (no round-half-away-
 * from-zero bias term) instead of the pinned rounding formula. Compiles,
 * and happens to match when scale_shift==0 (no shift to round), but is
 * guaranteed wrong whenever scale_shift>0 and there is a nonzero
 * remainder (true for the large majority of the swept elements). */
#include "kernel_api.h"
#include <stdint.h>

void candidate_kernel(const int32_t *raw, int16_t *out, int M, int N,
                      int32_t scale_mult, int scale_shift) {
    int total = M * N;
    for (int i = 0; i < total; i++) {
        int64_t r = (int64_t)raw[i] * (int64_t)scale_mult;
        int64_t q = (scale_shift > 0) ? (r >> scale_shift) : r;  /* WRONG: no rounding */
        if (q >  32767) q =  32767;
        if (q < -32768) q = -32768;
        out[i] = (int16_t)q;
    }
}
