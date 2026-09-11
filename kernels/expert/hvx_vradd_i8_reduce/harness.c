#include "harness_common.h"
#include "kernel_api.h"

#define G 106            /* 3*32 + 10 tail groups */
#define N (4 * G)        /* 424 int8 elements */

static int8_t a[N] HVX_ALIGN;
static int32_t out[G] HVX_ALIGN, ref[G] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x2ADD1u;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned edge-case groups. */
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;          /* group0 sum = 10 */
    a[4]=127; a[5]=127; a[6]=-128; a[7]=-128; /* group1 sum = -2 (extremes) */
    a[8]=-1; a[9]=1; a[10]=-1; a[11]=1;       /* group2 sum = 0 (cancellation) */
    { int base = 4*(G-1); a[base]=5; a[base+1]=5; a[base+2]=5; a[base+3]=5; } /* tail group sum=20 */

    for (int k = 0; k < G; k++) {
        int32_t sum = 0;
        for (int j = 0; j < 4; j++)
            sum += (int32_t)a[4*k+j];
        ref[k] = sum;
    }

    for (int k = 0; k < G; k++) out[k] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, G); });
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
