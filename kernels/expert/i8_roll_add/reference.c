#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int8_t *out, int n, int k) {
    for (int i = 0; i < n; i++)
        out[i] = (int8_t)(a[(i + k) % n] + b[i]);
}
