#include "harness_common.h"
#include "kernel_api.h"

#define N 780   /* 6*128 + 12 tail */

static uint8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN, wa[N] HVX_ALIGN, wb[N] HVX_ALIGN;
static int16_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00A9A2u;
    for (int i = 0; i < N; i++) {
        a[i]  = (uint8_t)(hvx_lcg(&s) >> 24);
        b[i]  = (uint8_t)(hvx_lcg(&s) >> 24);
        wa[i] = (uint8_t)(hvx_lcg(&s) >> 24);
        wb[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned wrap edge cases. */
    a[0]=200; wa[0]=200; b[0]=200; wb[0]=200;   /* 40000+40000=80000 -> wraps to 14464 */
    a[1]=255; wa[1]=255; b[1]=255; wb[1]=255;   /* 65025+65025=130050 -> wraps to -1022 */
    a[2]=0; wa[2]=0; b[2]=0; wb[2]=0;           /* 0 */
    a[3]=1; wa[3]=1; b[3]=0; wb[3]=255;         /* 1 + 0 = 1 (b term zeroed by a[3]=0... wait b=0*255=0) */
    a[N-1]=100; wa[N-1]=100; b[N-1]=100; wb[N-1]=100; /* 10000+10000=20000 (tail-path edge) */

    for (int i = 0; i < N; i++)
        ref[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i]);

    for (int i = 0; i < N; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, wa, wb, out, N); });
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
