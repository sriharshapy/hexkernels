#include "harness_common.h"
#include "kernel_api.h"

#define G 262             /* output-pair groups; tail path: 262 = 8*32 + 6 */
#define N (4 * G)         /* 1048 uint8 input elements */
#define OUTN (2 * G)      /* 524 int16 output elements */

static uint8_t a[N] HVX_ALIGN;
static int16_t out[OUTN] HVX_ALIGN, ref[OUTN] HVX_ALIGN;
static const int8_t w[4] = {127, 100, -128, 50};   /* fixed weight, shared across all groups */

int main(void) {
    uint32_t s = 0xD819u;
    for (int i = 0; i < N; i++)
        a[i] = (uint8_t)(hvx_lcg(&s) >> 24);

    /* Pinned edge-case groups. */
    /* Group 0: small known values -> out0=1*127+2*100=327, out1=3*(-128)+4*50=-184 */
    a[0]=1; a[1]=2; a[2]=3; a[3]=4;
    /* Group 1: large uint8 magnitudes -> forces int16 WRAP on out2 (57885 -> -7651) */
    a[4]=255; a[5]=255; a[6]=255; a[7]=255;
    /* Group 2: all-zero input -> out4=0, out5=0 */
    a[8]=0; a[9]=0; a[10]=0; a[11]=0;
    /* Last group (tail path, group G-1 = 261): out=5*127+5*100=1135, 5*(-128)+5*50=-390 */
    {
        int base = 4 * (G - 1);
        a[base]=5; a[base+1]=5; a[base+2]=5; a[base+3]=5;
    }

    for (int k = 0; k < G; k++) {
        int s0 = (int)a[4*k+0]*w[0] + (int)a[4*k+1]*w[1];
        int s1 = (int)a[4*k+2]*w[2] + (int)a[4*k+3]*w[3];
        ref[2*k]   = (int16_t)s0;
        ref[2*k+1] = (int16_t)s1;
    }

    for (int i = 0; i < OUTN; i++) {
        unsigned char *p = (unsigned char *)&out[i];
        p[0] = 0xA5; p[1] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, w, out, G); });
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
