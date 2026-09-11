/* Near-miss: off-by-one in row stride — uses (idx[i]+1)*E instead of idx[i]*E.
 * This reads the next row for every token, producing consistently wrong output
 * (except when idx[i] happens to point to a row identical to idx[i]+1, which
 * is random and nearly never true for distinct random table entries). */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int E) {
    /* BUG: off-by-one — reads row idx[i]+1 instead of idx[i] */
    for (int i = 0; i < T; i++) {
        int32_t row = idx[i] + 1;   /* BUG: +1 */
        const int8_t *src = table + (int)row * E;
        int8_t       *dst = out   + i * E;
        for (int e = 0; e < E; e++) dst[e] = src[e];
    }
}
