#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define N     512   /* output/bias length -- NOT a multiple of 128 */
#define N_SRC 256   /* gather source size; indices in [0, N_SRC) */

static int8_t  in_a[N_SRC] HVX_ALIGN;
static int8_t  in_b[N]     HVX_ALIGN;
static int32_t idx_buf[N]  HVX_ALIGN;
static int8_t  out[N]      HVX_ALIGN;
static int8_t  ref[N]      HVX_ALIGN;

/*
 * Two distinct (in_a, in_b, idx) sweeps so a kernel that ignores idx fails,
 * and one that ignores in_b also fails.
 */
#define NSETS 2

int main(void) {
    uint32_t s = 0x13579BDFu;

    int8_t  as  [NSETS][N_SRC];
    int8_t  bs  [NSETS][N];
    int32_t idxs[NSETS][N];

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N_SRC; i++)
            as[k][i] = (int8_t)(hvx_lcg(&s) >> 24);
        for (int i = 0; i < N; i++)
            bs[k][i] = (int8_t)(hvx_lcg(&s) >> 24);
        for (int i = 0; i < N; i++)
            idxs[k][i] = (int32_t)((hvx_lcg(&s) >> 16) % N_SRC);
    }

    /* Pinned edge cases in set 0 */
    idxs[0][0]   = 0;           /* first source element */
    idxs[0][1]   = N_SRC - 1;  /* last source element */
    idxs[0][2]   = 0;           /* repeated index (same gather, different bias) */
    idxs[0][N-1] = N_SRC / 2;  /* last output position */

    /* Set 1 uses reversed index pattern to ensure different data flow */
    for (int i = 0; i < N; i++)
        idxs[1][i] = idxs[0][N - 1 - i];

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < N_SRC; i++) in_a[i]   = as[k][i];
        for (int i = 0; i < N; i++)     in_b[i]   = bs[k][i];
        for (int i = 0; i < N; i++)     idx_buf[i] = idxs[k][i];

        /* Scalar reference: gather then add with int8 wraparound */
        for (int i = 0; i < N; i++)
            ref[i] = (int8_t)((int16_t)in_a[idx_buf[i]] + (int16_t)in_b[i]);

        /* Poison output */
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in_a, in_b, idx_buf, out, N); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < N; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb   = k * N + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
