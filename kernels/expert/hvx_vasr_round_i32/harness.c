#include "harness_common.h"
#include "kernel_api.h"

#define N 200      /* pairs; tail path: 200 = 6*32 + 8 (32 int32 lanes/vector) */
#define OUTN (2 * N)
#define SHIFT 5

static int32_t lo[N] HVX_ALIGN, hi[N] HVX_ALIGN;
static int16_t out[OUTN] HVX_ALIGN, ref[OUTN] HVX_ALIGN;

static int16_t round_shift_sat(int32_t x, int shift) {
    int32_t r = (shift == 0) ? x : ((x + (1 << (shift - 1))) >> shift);
    if (r > 32767) r = 32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

int main(void) {
    uint32_t s = 0xA53D1u;
    for (int i = 0; i < N; i++) {
        lo[i] = (int32_t)(hvx_lcg(&s) & 0x0FFFFFFF) - 0x08000000;  /* moderate range, no add-overflow */
        hi[i] = (int32_t)(hvx_lcg(&s) & 0x0FFFFFFF) - 0x08000000;
    }

    /* Pinned rounding/saturation edge cases (shift=5, divisor=32). */
    lo[0] = 16;          hi[0] = 100;          /* tie 0.5->1 ; normal 100/32~3 */
    lo[1] = -16;         hi[1] = 80;           /* tie -0.5->0 ; tie 2.5->3 */
    lo[2] = 2000000000;  hi[2] = -2000000000;  /* positive/negative SATURATION */
    lo[3] = 0;           hi[3] = -1;           /* zero; small negative */
    /* Tail-path edge: last pair (index N-1 = 199, inside the 8-elem tail). */
    lo[N-1] = 7; hi[N-1] = -7;

    for (int i = 0; i < N; i++) {
        ref[2*i]   = round_shift_sat(lo[i], SHIFT);
        ref[2*i+1] = round_shift_sat(hi[i], SHIFT);
    }

    for (int i = 0; i < OUTN; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(lo, hi, out, N, SHIFT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < OUTN; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, OUTN, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
