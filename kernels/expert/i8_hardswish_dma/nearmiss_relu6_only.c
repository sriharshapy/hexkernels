/* Near-miss: outputs relu6(x+3) (the gate) instead of the full hard-swish
 * x*relu6(x+3)/6. Matches only where the two happen to coincide; differs on
 * essentially all non-trivial inputs. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int v = (int)x[i];
        int r6 = v + 3; if (r6 < 0) r6 = 0; else if (r6 > 6) r6 = 6;
        out[i] = (int8_t)r6;      /* MISSING: * x / 6 */
    }
}
