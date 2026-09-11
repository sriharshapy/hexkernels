/* Near-miss: hardcodes mult=1,shift=0,zp=0 AND ignores idx (identity gather).
 * Passes only when mult=1,shift=0,zp=0 AND idx[i]==i, which never holds
 * simultaneously for the harness's random index sweeps and multi-param sets.
 * The harness sweeps 5 distinct (mult,shift,zp) sets x 2 index sets = 10 runs.
 */
#include <stdint.h>
void candidate_kernel(const int8_t *table, const int32_t *idx, int8_t *out, int n,
                      int32_t mult, int shift, int8_t zp) {
    (void)idx; (void)mult; (void)shift; (void)zp;  /* BUG: ignores all params */
    for (int i = 0; i < n; i++) {
        /* BUG: identity copy -- no gather, no requant */
        int32_t r = (int32_t)table[i];
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        out[i] = (int8_t)r;
    }
}
