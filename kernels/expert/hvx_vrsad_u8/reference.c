#include <stdint.h>
/* Scalar baseline: 4-wide SAD against fixed pattern {10,20,30,40}. */
static const int pattern[4] = {10, 20, 30, 40};
void candidate_kernel(const uint8_t *a, uint32_t *out, int g) {
    for (int k = 0; k < g; k++) {
        uint32_t sad = 0;
        for (int j = 0; j < 4; j++) {
            int d = (int)a[4*k+j] - pattern[j];
            sad += (uint32_t)(d < 0 ? -d : d);
        }
        out[k] = sad;
    }
}
