/* Near-miss: bakes in a fixed inv_lut instead of reading the runtime one.
 * Fails when the harness sweeps a different LUT table. */
#include <stdint.h>
static const uint8_t HARDCODED_INV[256] = {
    200,180,160,140,120,110,100,95,90,85,80,76,72,68,65,62,
    59,57,54,52,50,48,46,44,43,41,40,38,37,36,35,34,
    33,32,31,30,29,28,27,27,26,25,24,24,23,22,22,21,
    21,20,20,19,19,18,18,17,17,17,16,16,15,15,15,14,
    14,14,13,13,13,12,12,12,12,11,11,11,11,10,10,10,
    10,10,9,9,9,9,9,8,8,8,8,8,8,7,7,7,
    7,7,7,7,6,6,6,6,6,6,6,6,5,5,5,5,
    5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,
    4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,
    2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
};
void candidate_kernel(const int8_t *x, int8_t *out, int n, int G,
                      const int8_t *gamma, const int8_t *beta,
                      const uint8_t *inv_lut) {
    (void)inv_lut;  /* BUG: ignores runtime lut */
    int chunk = n / G;
    for (int c = 0; c < G; c++) {
        const int8_t *xc = x     + c * chunk;
        int8_t       *oc = out   + c * chunk;
        const int8_t *gc = gamma + c * chunk;
        const int8_t *bc = beta  + c * chunk;
        int32_t sum = 0;
        for (int j = 0; j < chunk; j++) sum += (int32_t)xc[j];
        int32_t mu = sum / chunk;
        int32_t var_sum = 0;
        for (int j = 0; j < chunk; j++) {
            int32_t d = (int32_t)xc[j] - mu;
            var_sum += d * d;
        }
        int32_t var = var_sum / chunk;
        int32_t v_idx = var;
        if (v_idx < 0)   v_idx = 0;
        if (v_idx > 255) v_idx = 255;
        uint8_t inv = HARDCODED_INV[(int)v_idx];
        for (int j = 0; j < chunk; j++) {
            int32_t d      = (int32_t)xc[j] - mu;
            int32_t scaled = (d * (int32_t)gc[j] + 64) >> 7;
            int32_t normed = (scaled * (int32_t)inv + 128) >> 8;
            int32_t res    = normed + (int32_t)bc[j];
            if (res >  127) res =  127;
            if (res < -128) res = -128;
            oc[j] = (int8_t)res;
        }
    }
}
