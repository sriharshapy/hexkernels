#include "kernel_api.h"
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n) {
    for (int i = 0; i < n; i++) {
        int s = (int)a[i] + (int)b[i];
        if (s > 127) s = 127;
        if (s < -128) s = -128;
        out[i] = (int8_t)s;
    }
}
