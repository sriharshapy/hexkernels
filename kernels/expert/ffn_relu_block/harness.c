#include "harness_common.h"
#include "kernel_api.h"
#define M  8    /* tokens */
#define K  16   /* input dim  */
#define V  32   /* hidden dim (ReLU layer) */
#define D  16   /* output dim */

static int8_t  A  [M*K]   HVX_ALIGN;
static int8_t  W1 [K*V]   HVX_ALIGN;
static int32_t b1 [V]     HVX_ALIGN;
static int8_t  W2 [V*D]   HVX_ALIGN;
static int32_t b2 [D]     HVX_ALIGN;
static int8_t  out[M*D]   HVX_ALIGN;
static int8_t  ref[M*D]   HVX_ALIGN;
static int8_t  hid[M*V]   HVX_ALIGN;

static int8_t sat8(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

/* Two requant param sets */
static const int32_t MULTS[]  = {  5,  2 };
static const int     SHIFTS[] = {  4,  3 };
static const int     ZPS[]    = {  0,  2 };
#define NSETS 2

int main(void) {
    uint32_t s = 0xB5E9D1u;
    for (int i = 0; i < M*K; i++) A [i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*V; i++) W1[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < V;   i++) b1[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < V*D; i++) W2[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D;   i++) b2[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases */
    A[0] = 127; W1[0] = 127;       /* large positive: relu passes through */
    A[K] = -128; W1[V] = 127;      /* sign edge */
    b1[0] = -200000;               /* large negative: relu clamps to 0 */
    b1[1] =  200000;               /* large positive: large hidden */
    /* Row 2: force negative acc+b1 so relu fires */
    for (int k = 0; k < K && k < 8; k++) {
        A[2*K+k] = -100;
        W1[k*V+2] = 100;
    }
    b1[2] = -100000;               /* acc+b1 < 0 -> relu -> 0 */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t mult = MULTS[p]; int shift = SHIFTS[p]; int8_t zp = (int8_t)ZPS[p];

        /* Reference layer 1: GEMM + b1 + ReLU + sat8 */
        for (int i = 0; i < M; i++) {
            for (int v = 0; v < V; v++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)W1[k*V+v];
                int64_t biased = (int64_t)acc + (int64_t)b1[v];
                if (biased < 0) biased = 0;
                hid[i*V+v] = sat8((int32_t)biased);
            }
        }

        /* Reference layer 2: GEMM + b2 + requant */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < D; j++) {
                int32_t acc = 0;
                for (int v = 0; v < V; v++)
                    acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
                int64_t biased = (int64_t)acc + (int64_t)b2[j];
                int64_t vv   = biased * (int64_t)mult;
                int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
                int64_t r    = (vv >= 0) ? ((vv + half) >> shift) : -(((-vv) + half) >> shift);
                r += zp;
                if (r >  127) r =  127;
                if (r < -128) r = -128;
                ref[i*D+j] = (int8_t)r;
            }
        }

        /* Poison output */
        for (int i = 0; i < M*D; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, W1, b1, W2, b2, out, M, K, V, D, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < M*D; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) { fb = p*M*D + i; gotv = (long)out[i]; expv = (long)ref[i]; }
            }
        }
    }
    hvx_report(errors, M*D*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
