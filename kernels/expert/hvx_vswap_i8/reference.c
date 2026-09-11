#include <stdint.h>
/* Scalar baseline: per-lane conditional swap-select. */
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out_hi, int8_t *out_lo, int n) {
    for (int i = 0; i < n; i++) {
        int pred = a[i] > b[i];
        out_hi[i] = pred ? a[i] : b[i];
        out_lo[i] = pred ? b[i] : a[i];
    }
}
