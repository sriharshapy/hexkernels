/* Baseline: scalar matmul + HMX 0x40 requant field + saturating narrow to int8. */
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int f) { int v = f & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }

void candidate_kernel(const uint8_t *A, const int8_t *B, int8_t *out, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            int acc = 0;
            for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)B[k*n+j];
            int r = sx12((acc * 17 + 8) >> 4);
            if (r > 127) r = 127; else if (r < -128) r = -128;
            out[i*n + j] = (int8_t)r;
        }
}
