/* Near-miss: returns the LAST occurrence of the minimum (uses <=), so on the
 * injected tie it yields the later index instead of the first. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int n, int32_t *out) {
    int bi = 0, bv = (int)a[0];
    for (int i = 1; i < n; i++) if ((int)a[i] <= bv) { bv = a[i]; bi = i; }
    out[0] = bi;
}
