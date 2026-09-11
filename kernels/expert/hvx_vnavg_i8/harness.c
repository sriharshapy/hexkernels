#include "harness_common.h"
#include "kernel_api.h"

#define N 1091   /* 8*128 + 67 tail */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x00CA46u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases (empirically verified against Q6_Vb_vnavg_VbVb). */
    a[0] = 10;  b[0] = 4;      /* (10-4)>>1 = 3 */
    a[1] = -10; b[1] = 4;      /* (-10-4)>>1 = -7 */
    a[2] = 5;   b[2] = 5;      /* (5-5)>>1 = 0 */
    a[3] = 5;   b[3] = -3;     /* (5+3)>>1 = 4 */
    a[4] = 127; b[4] = -128;   /* (255)>>1 = 127 */
    a[5] = -128; b[5] = 127;   /* (-255)>>1 = -128 */
    a[N-1] = -128; b[N-1] = -128; /* tail-path edge: (0)>>1=0 */

    for (int i = 0; i < N; i++)
        ref[i] = (int8_t)(((int)a[i] - (int)b[i]) >> 1);

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

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
