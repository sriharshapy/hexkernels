/* Baseline: pure scalar top-1 reduction (max value + first index of that value).
 * Matches the harness's golden computation exactly: scan left-to-right, keep the
 * running max and its first index, strict '>' comparison so ties keep the
 * earliest index. */
#include <stdint.h>
#include "kernel_api.h"

void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int bv = (int)a[0];
    int bi = 0;
    for (int i = 1; i < n; i++) {
        if ((int)a[i] > bv) { bv = a[i]; bi = i; }
    }
    out[0] = (int32_t)bv;
    out[1] = (int32_t)bi;
}
