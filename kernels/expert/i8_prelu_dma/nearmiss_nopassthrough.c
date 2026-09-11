/* Near-miss: applies the alpha/shift scale to ALL elements (drops the x>0
 * passthrough), so every positive input is wrongly scaled down. */
#include <stdint.h>
void candidate_kernel(const int8_t *x, int8_t *out,
                      int n_ch, int n_elem,
                      const int8_t *alpha, int shift) {
    for (int c = 0; c < n_ch; c++) {
        int a = (int)alpha[c];
        for (int i = 0; i < n_elem; i++) {
            int r = ((int)x[c*n_elem+i] * a) >> shift;
            if (r > 127) r = 127; if (r < -128) r = -128;
            out[c*n_elem+i] = (int8_t)r;
        }
    }
}
