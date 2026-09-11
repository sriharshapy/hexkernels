#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define N       512   /* output length -- NOT a multiple of 128 */
#define N_TABLE 256   /* lookup table size; indices in [0, N_TABLE) */

static int8_t  table[N_TABLE] HVX_ALIGN;
static int32_t idx_buf[N]     HVX_ALIGN;
static int8_t  out[N]         HVX_ALIGN;
static int8_t  ref[N]         HVX_ALIGN;

static int8_t ref_element(int8_t raw8, int32_t mult, int shift, int8_t zp) {
    int32_t raw = (int32_t)raw8;
    int64_t v    = (int64_t)raw * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift)
                             : -(((-v) + half) >> shift);
    r += (int64_t)zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/*
 * Multiple parameter sweeps: candidate MUST read mult/shift/zp at runtime.
 * Multiple index sweeps: candidate MUST use the runtime idx pointer.
 * Combinations: NPARAMS param sets x NIDX index sets = NSETS total runs.
 */
static const int32_t MULTS[]  = {  1,  5, -3, 13,  1 };
static const int     SHIFTS[] = {  0,  4,  2,  7,  3 };
static const int     ZPS[]    = {  0,  0, 10, -5,  2 };
#define NPARAMS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))
#define NIDX    2   /* distinct index sweeps per param set */

int main(void) {
    uint32_t s = 0x2468ACEu;

    int8_t  tables[NIDX][N_TABLE];
    int32_t idxs  [NIDX][N];

    for (int k = 0; k < NIDX; k++) {
        for (int i = 0; i < N_TABLE; i++)
            tables[k][i] = (int8_t)(hvx_lcg(&s) >> 24);
        for (int i = 0; i < N; i++)
            idxs[k][i] = (int32_t)((hvx_lcg(&s) >> 16) % N_TABLE);
    }

    /* Pinned edge cases for index set 0 */
    idxs[0][0]   = 0;           /* first table entry */
    idxs[0][1]   = N_TABLE - 1; /* last table entry */
    idxs[0][2]   = 0;           /* repeated index */
    idxs[0][N-1] = N_TABLE / 2; /* last output position */

    /* Index set 1 uses the reverse of index set 0 */
    for (int i = 0; i < N; i++)
        idxs[1][i] = idxs[0][N - 1 - i];

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int p = 0; p < NPARAMS; p++) {
        int32_t mult  = MULTS[p];
        int     shift = SHIFTS[p];
        int8_t  zp    = (int8_t)ZPS[p];

        for (int k = 0; k < NIDX; k++) {
            for (int i = 0; i < N_TABLE; i++) table[i]   = tables[k][i];
            for (int i = 0; i < N; i++)       idx_buf[i] = idxs[k][i];

            /* Scalar reference */
            for (int i = 0; i < N; i++)
                ref[i] = ref_element(table[idx_buf[i]], mult, shift, zp);

            /* Poison output */
            for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

            unsigned long long _hvx_kc = 0;
            HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx_buf, out, N, mult, shift, zp); });
            printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

            for (int i = 0; i < N; i++) {
                if (out[i] != ref[i]) {
                    errors++;
                    if (fb < 0) {
                        fb   = (p * NIDX + k) * N + i;
                        gotv = (long)out[i];
                        expv = (long)ref[i];
                    }
                }
            }
        }
    }

    hvx_report(errors, N * NPARAMS * NIDX, fb, gotv, expv);
    return errors ? 1 : 0;
}
