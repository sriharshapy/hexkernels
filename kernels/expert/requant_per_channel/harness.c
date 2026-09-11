#include "harness_common.h"
#include "kernel_api.h"

/* Fixed shape: 40 rows x 25 columns = 1000 elements total. */
#define R 40
#define C 25
#define N (R*C)

static int32_t a[N]   HVX_ALIGN;
static int8_t  out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

/* Per-channel mult/shift arrays — 5 sweeps, each has R=40 entries.
   Sweeps include: negative mult (row 3), shift==0 (row 1), large mult (row 4),
   varied zero-points. This forces the candidate to READ mult[] and shift[] arrays
   (a hardcoded scalar fails immediately on any non-first-sweep). */
static const int32_t MULT_SETS[5][R] = {
    /* set 0: mostly 5, one negative, one large */
    { 5, 5, 5,-3, 127, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 },
    /* set 1: all 13 */
    {13,13,13,13, 13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,
     13,13,13,13, 13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13 },
    /* set 2: alternating 7 / -7 */
    { 7,-7, 7,-7,  7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7,
      7,-7, 7,-7,  7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7, 7,-7 },
    /* set 3: small values 1..4 cycling */
    { 1, 2, 3, 4,  1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4,
      1, 2, 3, 4,  1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 },
    /* set 4: powers-of-2 cycling */
    { 1, 2, 4, 8, 16, 1, 2, 4, 8,16, 1, 2, 4, 8,16, 1, 2, 4, 8,16,
      1, 2, 4, 8, 16, 1, 2, 4, 8,16, 1, 2, 4, 8,16, 1, 2, 4, 8,16 },
};
static const int SHIFT_SETS[5][R] = {
    /* set 0 */
    { 3, 0, 7, 2,  4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    /* set 1 */
    { 7, 7, 7, 7,  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
      7, 7, 7, 7,  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 },
    /* set 2 */
    { 3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
      3, 3, 3, 3,  3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 },
    /* set 3 */
    { 0, 1, 2, 3,  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3,
      0, 1, 2, 3,  0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3 },
    /* set 4 */
    { 0, 1, 2, 3,  4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4,
      0, 1, 2, 3,  4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4 },
};
static const int8_t ZPS[5] = { 0, -5, 0, 10, 0 };
#define NSETS 5

static int8_t requant_ref(int32_t x, int32_t mult, int shift, int8_t zp) {
    int64_t v = (int64_t)x * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r > 127) r = 127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

int main(void) {
    uint32_t s = 0xBEEF77u;
    for (int i = 0; i < N; i++) a[i] = (int32_t)((int32_t)(hvx_lcg(&s)) % 601) - 300;
    /* inject boundary values across first few rows */
    a[0] = 4; a[1] = -4; a[2] = 300; a[3] = -300; a[4] = 0;
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int k = 0; k < NSETS; k++) {
        const int32_t *mult = MULT_SETS[k];
        const int     *shift = SHIFT_SETS[k];
        int8_t         zp   = ZPS[k];
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                ref[r*C+c] = requant_ref(a[r*C+c], mult[r], shift[r], zp);
        for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, R, C, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i = 0; i < N; i++)
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = k * N + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
    }
    hvx_report(errors, N * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
