#include "harness_common.h"
#include "kernel_api.h"

#define N 519   /* 8*64 + 7 tail */

static int16_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

static int16_t vmpyh_ref(int16_t av, int16_t bv) {
    int32_t P = (int32_t)av * (int32_t)bv;
    int32_t bias = (P >= 0) ? 16384 : -16384;
    int32_t q = (P + bias) / 32768;
    if (q > 32767) q = 32767;
    if (q < -32768) q = -32768;
    return (int16_t)q;
}

int main(void) {
    uint32_t s = 0x00A9DEu;
    for (int i = 0; i < N; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    /* Pinned edge cases (empirically verified against Q6_Vh_vmpy_VhVh_s1_rnd_sat). */
    a[0] = 16384; b[0] = 16384;              /* 0.5*0.5 -> 8192 (0.25) */
    a[1] = 16384; b[1] = (int16_t)0xC000;    /* 0.5*-0.5 -> -8192 */
    a[2] = (int16_t)0x8000; b[2] = (int16_t)0x8000; /* -1*-1 -> sat 32767 */
    a[3] = 32767; b[3] = 32767;              /* ~1*1 -> 32766 (not-quite-1 rounding) */
    a[4] = 1; b[4] = 1;                      /* tiny -> 0 */
    a[N-1] = 100; b[N-1] = 200;              /* tail-path edge -> 1 */

    for (int i = 0; i < N; i++)
        ref[i] = vmpyh_ref(a[i], b[i]);

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
