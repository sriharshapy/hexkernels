#include "harness_common.h"
#include "kernel_api.h"
#define N 64   /* S == Din == Dout == 64 */

static uint8_t  X[N*N]   HVX_ALIGN;
static int8_t   W[N*N]   HVX_ALIGN;
static int32_t  bias[N]  HVX_ALIGN;
static int8_t   out[N*N] HVX_ALIGN;
static int8_t   ref[N*N] HVX_ALIGN;

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

static inline int8_t saturate_i8(int v) {
    if (v >  127) v =  127;
    if (v < -128) v = -128;
    return (int8_t)v;
}

int main(void) {
    uint32_t s = 0x3D8Cu;
    /* X in 0..7, W in -3..3: same bound analysis as i8_matmul_hmx_64x64
     * (|acc|<=64*7*3=1344 -> |r|<=1428). Bias spans +-2000 so the add crosses
     * the int8 saturation boundary on both sides. */
    for (int i = 0; i < N*N; i++) X[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < N*N; i++) W[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int j = 0; j < N;   j++) bias[j] = (int32_t)((int)(hvx_lcg(&s) % 4001) - 2000);
    bias[0] = -2000; bias[1] = 2000; bias[2] = 0; /* forces sat-low, sat-high, passthrough */

    for (int i = 0; i < N; i++)
        for (int o = 0; o < N; o++) {
            int acc = 0;
            for (int d = 0; d < N; d++) acc += (int)X[i*N+d] * (int)W[o*N+d];
            int r = sx12((acc * 17 + 8) >> 4);
            int biased = r + bias[o];
            ref[i*N+o] = saturate_i8(biased);
        }

    for (int i = 0; i < N*N; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, W, bias, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < N*N; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, N*N, fb, gotv, expv);
    return errors ? 1 : 0;
}
