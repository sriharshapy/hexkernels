#include "harness_common.h"
#include "kernel_api.h"

#define T_DIM  4
#define L_DIM  256
#define N      (T_DIM * L_DIM)   /* 1024 total elements */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xC0FFEEu;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge cases: overflow discriminators (wrap, not saturate). */
    /* Tensor 0, element 0: 127 + 127 = -2 (not 127) */
    a[0] = 127;  b[0] = 127;
    /* Tensor 0, element 1: 100 + 100 = -56 */
    a[1] = 100;  b[1] = 100;
    /* Tensor 0, element 2: -128 + (-1) = 127 */
    a[2] = -128; b[2] = -1;
    /* Tensor 0, element 3: 127 + 1 = -128 */
    a[3] = 127;  b[3] = 1;
    /* Last tensor's last element: tensor 3, element L-1 = index 1023 */
    a[N-1] = 50; b[N-1] = 50;

    /* Scalar reference */
    for (int i = 0; i < N; i++)
        ref[i] = (int8_t)((int)a[i] + (int)b[i]);

    /* Poison output */
    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, T_DIM, L_DIM); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, N, fb,
               fb >= 0 ? (long)(unsigned char)out[fb] : 0L,
               fb >= 0 ? (long)(unsigned char)ref[fb] : 0L);
    return errors ? 1 : 0;
}
