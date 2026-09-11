#include "harness_common.h"
#include "kernel_api.h"

#define N 900   /* 7*128 + 4 tail */

static int8_t a[N] HVX_ALIGN;
static int8_t out[N] HVX_ALIGN, ref[N] HVX_ALIGN;
static int8_t lut[256] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x1FF80Cu;
    for (int i = 0; i < N; i++)
        a[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < 256; i++)
        lut[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Pinned edge cases: known table entries + known indices. */
    lut[0] = 11; lut[1] = 22; lut[127] = 33; lut[128] = -44; lut[255] = -55;
    a[0] = 0;            /* idx 0   -> lut[0]=11 */
    a[1] = 1;             /* idx 1   -> lut[1]=22 */
    a[2] = 127;           /* idx 127 -> lut[127]=33 */
    a[3] = (int8_t)128;   /* bit pattern 0x80 -> as int8_t this IS -128; idx=(uint8_t)(-128)=128 -> lut[128]=-44 */
    a[N-1] = -1;          /* bit pattern 0xFF -> idx=255 -> lut[255]=-55 (tail-path edge) */

    for (int i = 0; i < N; i++) {
        uint8_t idx = (uint8_t)a[i];
        ref[i] = lut[idx];
    }

    for (int i = 0; i < N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, lut); });
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
