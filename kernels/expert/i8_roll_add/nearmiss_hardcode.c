/* Near-miss: hardcodes k=1, ignoring the runtime k parameter.
   Fails when harness sweeps k=128, 300, 1023. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k) {
    (void)k;
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(a[(i + 1) % n] + b[i]);
}
