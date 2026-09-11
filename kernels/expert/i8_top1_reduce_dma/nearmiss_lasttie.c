/* Near-miss: correct max value but LAST occurrence index (uses >=), so out[1] is
 * the later tie index instead of the first. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int bv = (int)a[0], bi = 0;
    for (int i = 1; i < n; i++) if ((int)a[i] >= bv) { bv = a[i]; bi = i; }
    out[0] = bv;
    out[1] = bi;
}
