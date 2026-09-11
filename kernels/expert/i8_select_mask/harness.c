#include "harness_common.h"
#include "kernel_api.h"
#define N 1000
static int8_t  a[N]    HVX_ALIGN;
static int8_t  b[N]    HVX_ALIGN;
static uint8_t mask[N] HVX_ALIGN;
static int8_t  out[N]  HVX_ALIGN;
static int8_t  ref[N]  HVX_ALIGN;

int main(void) {
    uint32_t s = 0x7B3F21u;
    for (int i = 0; i < N; i++) {
        uint32_t r = hvx_lcg(&s);
        a[i]    = (int8_t)(r >> 24);
        b[i]    = (int8_t)(r >> 16);
        /* mask: 0 or non-zero (use 0xFF for true, 0 for false, and some 0x01 for nonzero-true). */
        uint8_t m = (uint8_t)(r >> 8);
        mask[i] = (m < 85) ? 0 : (m < 170) ? 0xFF : 0x01;
    }
    /* Inject explicit cases: mask==0 must pick b, mask!=0 (0xFF and 0x01) must pick a. */
    mask[0] = 0;    a[0] = 42;  b[0] = -13;   /* expect b[0]=-13 */
    mask[1] = 0xFF; a[1] = 99;  b[1] = -99;   /* expect a[1]=99 */
    mask[2] = 0x01; a[2] = -1;  b[2] = 1;     /* nonzero=1 → expect a[2]=-1 */
    mask[3] = 0;    a[3] = -128; b[3] = 127;  /* min/max edge → expect b[3]=127 */
    /* Nonzero tail: elements [128*7..999] must still be processed. */

    for (int i = 0; i < N; i++) ref[i] = mask[i] ? a[i] : b[i];
    for (int i = 0; i < N; i++) out[i] = 0x5A;  /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, mask, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)(uint8_t)out[i]; expv = (long)(uint8_t)ref[i]; }
        }
    }
    hvx_report(errors, N, fb, gotv, expv);
    return errors ? 1 : 0;
}
