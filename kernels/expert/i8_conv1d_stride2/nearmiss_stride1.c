/* Near-miss: uses stride 1 (ignores the stride parameter), producing n outputs
   from consecutive samples instead of every-other-sample. Compiles and runs but
   gives wrong results for stride=2. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                      int n, int ntaps, int stride) {
    (void)stride;  /* BUG: stride ignored, always uses 1 */
    for (int i = 0; i < n; i++) {
        int32_t acc = 0;
        for (int j = 0; j < ntaps; j++)
            acc += (int32_t)x[i + j] * (int32_t)taps[j];
        out[i] = acc;
    }
}
