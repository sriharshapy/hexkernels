/* attention_scale_i16 harness. Elementwise int32->int16 fixed-point rescale
 * (round-half-away-from-zero), swept over 3 (scale_mult, scale_shift) sets.
 * Harness owns main() and computes the scalar reference INDEPENDENTLY. */
#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define M 16
#define N 37     /* NOT a multiple of 32 int32-lanes-per-HVX-vector */
#define TOTAL (M*N)   /* 592 = 18*32 + 16 (real tail) */

static int32_t raw[TOTAL] HVX_ALIGN;
static int16_t out[TOTAL] HVX_ALIGN;
static int16_t ref[TOTAL] HVX_ALIGN;

static int16_t rescale_ref(int32_t r_in, int32_t mult, int shift) {
    int64_t r    = (int64_t)r_in * (int64_t)mult;
    int64_t half = (shift > 0) ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t q    = (r >= 0) ? ((r + half) >> shift) : -((-r + half) >> shift);
    if (q >  32767) q =  32767;
    if (q < -32768) q = -32768;
    return (int16_t)q;
}

/* Param sweep: (scale_mult, scale_shift). Set 0 approximates 1/sqrt(64)
 * exactly (1/8 = 0.125). Set 1 approximates 1/sqrt(128) (181/2048 =
 * 0.08838, vs true 0.08839). Set 2 has scale_shift=0 (no rounding bias)
 * and a large mult so ordinary-magnitude inputs already saturate.
 * scale_mult is always POSITIVE (sign of q always matches sign of raw). */
static const int32_t MULTS[]  = {   1,  181, 500 };
static const int     SHIFTS[] = {   3,   11,   0 };
#define NSETS 3

int main(void) {
    uint32_t s = 0x37A11E5u;

    /* |raw| <= 1,000,000 so raw*mult (mult<=500) stays well within int32/int64
     * range for any implementation (no native HVX 32x32->64 widen needed). */
    for (int i = 0; i < TOTAL; i++)
        raw[i] = (int32_t)((int32_t)(hvx_lcg(&s) % 2000001) - 1000000);

    /* Pinned edge cases (flat indices; op is purely elementwise). */
    raw[0] = 2000000;   /* near safe-bound ceiling, positive -- saturates set 2 */
    raw[1] = -2000000;  /* near ceiling, negative -- saturates set 2 */
    raw[2] = 0;         /* zero */
    raw[3] = 1;         /* small positive -- discriminates round vs truncate */
    raw[4] = -1;        /* small negative */
    raw[5] = 4;         /* rounding TIE for set 0 (mult=1,shift=3): r=4, half=4 -> (4+4)>>3=1 */
    raw[6] = -4;        /* symmetric negative tie for set 0 */
    /* Tail-path edge (592 = 18*32 + 16; last 16 elements are the tail). */
    raw[TOTAL-2] = 3000000;   /* large positive in the tail -- saturates set 2 */
    raw[TOTAL-1] = -3000000;  /* large negative in the tail -- saturates set 2 */

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        int32_t mult  = MULTS[k];
        int     shift = SHIFTS[k];

        for (int i = 0; i < TOTAL; i++) ref[i] = rescale_ref(raw[i], mult, shift);
        for (int i = 0; i < TOTAL; i++) out[i] = (int16_t)0xA5A5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(raw, out, M, N, mult, shift); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < TOTAL; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * TOTAL + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, TOTAL * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
