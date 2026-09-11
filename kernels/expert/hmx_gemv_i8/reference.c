/* Scalar int8 GEMV (bit-exact) — the DENOMINATOR baseline. Plain per-row dot
 * product (A[i][:] . x) accumulated with a scalar loop, then the 0x40 requant.
 * The HMX matrix engine (which must fill a full 32x32 tile) is over-
 * provisioned for N=1; this naive scalar loop is the speedup denominator. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *A, const int8_t *x, int32_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int acc = 0;
        for (int k = 0; k < n; k++) acc += (int)A[i*n+k] * (int)x[k];
        out[i] = sx12((acc * 17 + 8) >> 4);
    }
}
