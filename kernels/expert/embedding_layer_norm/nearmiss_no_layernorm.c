/* Near-miss: performs the gather correctly but skips LayerNorm —
 * outputs the raw embedding values without normalization. */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out,
                      int T, int D,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)gamma;   /* BUG: normalization skipped */
    (void)beta;
    (void)inv_lut;
    /* BUG: just gather, no LayerNorm */
    for (int i = 0; i < T; i++) {
        const int8_t *row = table + (int)idx[i] * D;
        int8_t *out_row   = out + i * D;
        for (int j = 0; j < D; j++) out_row[j] = row[j];
    }
}
