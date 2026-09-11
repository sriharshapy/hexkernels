/* Near-miss: forgets to subtract the zero-point before scaling. Correct only
 * when zp == 0; here zp != 0 so it differs on essentially every element. */
#include <stdint.h>
void candidate_kernel(const int8_t *a, int16_t *out, int n,
                      int8_t zp, int32_t scale, int shift) {
    (void)zp;
    for (int i = 0; i < n; i++) {
        int32_t v = (int32_t)a[i] * scale;   /* MISSING: - zp */
        int32_t r = v >> shift;
        if (r > 32767)  r = 32767;
        if (r < -32768) r = -32768;
        out[i] = (int16_t)r;
    }
}
