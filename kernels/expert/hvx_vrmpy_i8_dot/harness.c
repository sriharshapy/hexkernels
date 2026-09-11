#include "harness_common.h"
#include "kernel_api.h"

#define G 261            /* output groups; tail path: 261 = 8*32 + 5 */
#define N (4 * G)        /* 1044 int8 elements per input */

static int8_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int32_t out[G] HVX_ALIGN, ref[G] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xB219u;
    for (int i = 0; i < N; i++) {
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
        b[i] = (int8_t)(hvx_lcg(&s) >> 24);
    }

    /* Pinned edge-case groups (signed dot product, no saturation). */
    /* Group 0: 1*10+2*20+3*30+4*40 = 300 */
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;      b[0]=10; b[1]=20; b[2]=30; b[3]=40;
    /* Group 1: negative + extreme magnitudes: -1-2+127*1-128*1 = -4 */
    a[4]=-1; a[5]=-2; a[6]=127; a[7]=-128;  b[4]=1; b[5]=1; b[6]=1; b[7]=1;
    /* Group 2: zero dot (orthogonal-ish cancellation): 1*1 + 1*(-1) + 0*5 + 0*(-5) = 0 */
    a[8]=1; a[9]=1; a[10]=0; a[11]=0;    b[8]=1; b[9]=-1; b[10]=5; b[11]=-5;
    /* Last group (tail path, group G-1 = 260): 2*3+2*3+2*3+2*3 = 24 */
    {
        int base = 4 * (G - 1);
        a[base]=2; a[base+1]=2; a[base+2]=2; a[base+3]=2;
        b[base]=3; b[base+1]=3; b[base+2]=3; b[base+3]=3;
    }

    for (int k = 0; k < G; k++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++)
            sum += (int32_t)a[4*k+j] * (int32_t)b[4*k+j];
        ref[k] = sum;
    }

    for (int k = 0; k < G; k++) out[k] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, G); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int k = 0; k < G; k++) {
        if (out[k] != ref[k]) {
            errors++;
            if (fb < 0) fb = k;
        }
    }
    hvx_report(errors, G, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
