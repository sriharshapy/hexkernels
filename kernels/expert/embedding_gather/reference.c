#include <stdint.h>
/*
 * Embedding gather baseline.
 * out[i*E + e] = table[idx[i]*E + e]   for i in [0,T), e in [0,E)
 */
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E) {
    for (int i = 0; i < T; i++) {
        int32_t row = idx[i];
        const int8_t *src = table + (int)row * E;
        int8_t       *dst = out   + i * E;
        for (int e = 0; e < E; e++) {
            dst[e] = src[e];
        }
    }
}
