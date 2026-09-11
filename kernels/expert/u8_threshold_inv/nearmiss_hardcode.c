/* Near-miss: hardcodes thresh=128, ignoring the runtime param.
   Fails because the harness sweeps several thresholds (anti-hardcode gate). */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int n, uint8_t thresh) {
    (void)thresh;
    for (int i = 0; i < n; i++) out[i] = (in[i] < 128) ? 255 : 0;
}
