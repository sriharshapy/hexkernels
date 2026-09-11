#include "harness_common.h"
#include "kernel_api.h"

#define N 210   /* 3*64 + 18 tail (pairs) */

static int16_t a[N] HVX_ALIGN, b[N] HVX_ALIGN;
static int8_t out[2*N] HVX_ALIGN, ref[2*N] HVX_ALIGN;

static int8_t sat8(int v) {
    if (v > 127) return 127;
    if (v < -128) return -128;
    return (int8_t)v;
}
static int round_div256(int16_t x) {
    return ((int)x + 128) >> 8;
}

int main(void) {
    uint32_t s = 0x00B0DEu;
    for (int i = 0; i < N; i++) {
        a[i] = (int16_t)(hvx_lcg(&s) >> 16);
        b[i] = (int16_t)(hvx_lcg(&s) >> 16);
    }

    /* Pinned rounding-tie + saturation edge cases. */
    a[0] = 128;             /* tie: (128+128)>>8=1 -> out[1]=1 */
    a[1] = 384;             /* tie: 512>>8=2 -> out[3]=2 */
    a[2] = (int16_t)0xFF80; /* -128 tie: 0>>8=0 -> out[5]=0 */
    a[3] = 32767;           /* saturates: (32895)>>8=128 -> clamp 127 -> out[7]=127 */
    a[4] = (int16_t)0x8000; /* -32768: (-32640)>>8=-128 (exact, no extra clamp) -> out[9]=-128 */
    b[0] = 5000;            /* (5128)>>8=20 -> out[0]=20 */
    b[N-1] = -1003;         /* tail-path edge: (-875)>>8=-4 -> out[2N-2]=-4 */

    for (int i = 0; i < N; i++) {
        ref[2*i]     = sat8(round_div256(b[i]));
        ref[2*i + 1] = sat8(round_div256(a[i]));
    }

    for (int i = 0; i < 2*N; i++) out[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < 2*N; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) fb = i;
        }
    }
    hvx_report(errors, 2*N, fb,
               fb >= 0 ? (long)out[fb] : 0L,
               fb >= 0 ? (long)ref[fb] : 0L);
    return errors ? 1 : 0;
}
