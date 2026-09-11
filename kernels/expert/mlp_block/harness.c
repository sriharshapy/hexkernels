#include "harness_common.h"
#include "kernel_api.h"
/* Dims budget: two GEMMs O(d^2). Keep tiny so sim finishes well under 60 s. */
#define M  8    /* tokens */
#define K  16   /* input dim  */
#define V  32   /* hidden dim (GELU layer) */
#define D  16   /* output dim */

static int8_t  A  [M*K]   HVX_ALIGN;
static int8_t  W1 [K*V]   HVX_ALIGN;
static int32_t b1 [V]     HVX_ALIGN;
static int8_t  gelu_lut[256] HVX_ALIGN;
static int8_t  W2 [V*D]   HVX_ALIGN;
static int32_t b2 [D]     HVX_ALIGN;
static int8_t  out[M*D]   HVX_ALIGN;
static int8_t  ref[M*D]   HVX_ALIGN;
/* Intermediate hidden (int8 after GELU) */
static int8_t  hid[M*V]   HVX_ALIGN;

/* Build GELU LUT variant */
static void build_gelu_lut(int8_t *lut, int variant) {
    for (int i = 0; i < 256; i++) {
        int v = i - 128;
        float fv = (float)v;
        if (variant == 1) fv *= 0.85f;
        float t = 0.7978845608f * (fv + 0.044715f * fv * fv * fv);
        float th;
        if      (t >  4.0f) th =  1.0f;
        else if (t < -4.0f) th = -1.0f;
        else { float x2 = t*t; th = t*(27.0f + x2)/(27.0f + 9.0f*x2); }
        float gv = 0.5f * fv * (1.0f + th);
        int r = (int)(gv >= 0 ? gv + 0.5f : gv - 0.5f);
        if (r >  127) r =  127;
        if (r < -128) r = -128;
        lut[i] = (int8_t)r;
    }
}

static int8_t sat8(int32_t x) {
    if (x >  127) return  127;
    if (x < -128) return -128;
    return (int8_t)x;
}

static int8_t requant_ref(int64_t v32, int32_t mult, int shift, int8_t zp) {
    int64_t v    = v32 * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

/* Two requant param sets (anti-hardcode) */
static const int32_t MULTS[]  = {  3,  7 };
static const int     SHIFTS[] = {  4,  5 };
static const int     ZPS[]    = {  0, -3 };
#define NSETS 2

int main(void) {
    uint32_t s = 0xABC123u;
    for (int i = 0; i < M*K; i++) A [i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < K*V; i++) W1[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < V;   i++) b1[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);
    for (int i = 0; i < V*D; i++) W2[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < D;   i++) b2[i] = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases:
     * - Large product in layer 1: acc saturates before LUT
     * - Negative pre-activation in layer 1 (GELU suppresses)
     * - Large negative b1 to force sat to -128 -> negative GELU region
     * - Large positive b1 to force sat to +127
     */
    A[0] = 127; W1[0] = 127;      /* large positive acc */
    A[K] = -128; W1[V] = 127;     /* sign edge */
    b1[0] = -200000;               /* large negative bias -> sat to -128 -> GELU near 0 */
    b1[1] =  200000;               /* large positive bias -> sat to +127 */
    /* Force negative pre-activation in layer 1 row 2 */
    for (int k = 0; k < K && k < 8; k++) {
        A[2*K+k] = -100;
        W1[k*V+2] = 100;
    }
    b1[2] = -100000;               /* ensure acc+b1 < 0 -> GELU suppresses */

    int errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        build_gelu_lut(gelu_lut, p);
        int32_t mult = MULTS[p]; int shift = SHIFTS[p]; int8_t zp = (int8_t)ZPS[p];

        /* --- Reference layer 1: GEMM + b1 + sat + GELU LUT --- */
        for (int i = 0; i < M; i++) {
            for (int v = 0; v < V; v++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)A[i*K+k] * (int32_t)W1[k*V+v];
                acc += b1[v];
                int8_t pre = sat8(acc);
                uint8_t idx = (uint8_t)(pre + 128);
                hid[i*V+v] = gelu_lut[idx];
            }
        }

        /* --- Reference layer 2: GEMM + b2 + requant --- */
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < D; j++) {
                int32_t acc = 0;
                for (int v = 0; v < V; v++)
                    acc += (int32_t)hid[i*V+v] * (int32_t)W2[v*D+j];
                int64_t biased = (int64_t)acc + (int64_t)b2[j];
                ref[i*D+j] = requant_ref(biased, mult, shift, zp);
            }
        }

        /* Poison output */
        for (int i = 0; i < M*D; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, W1, b1, gelu_lut, W2, b2, out, M, K, V, D, mult, shift, zp); });
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
