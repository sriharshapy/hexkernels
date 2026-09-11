/* Near-miss: VALID correlation (no padding), giving only n-ntaps+1=994 outputs
   but storing them at out[0..993], leaving out[994..999] unpoisoned/wrong.
   Also wrong for the first pad_left outputs which should use zero-padding.
   Compiles cleanly but fails the harness (boundary mismatches). */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps) {
    /* BUG: no padding — treats x as already having ntaps-1 extra samples,
       which it doesn't (x has exactly n elements). Accesses x[0..n-1] only
       for i=0 because inner j goes to ntaps-1, reading x[ntaps-1] at i=0,
       which is in-bounds, but out[0] misses the zero-pad contribution.
       Worse, out[n-ntaps+1 .. n-1] are never written (left poisoned). */
    int valid_n = n - ntaps + 1;
    for (int i = 0; i < valid_n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++)
            acc += (int32_t)x[i + j] * (int32_t)taps[j];
        out[i] = acc;
    }
    /* out[valid_n .. n-1] remain at poison value */
}
