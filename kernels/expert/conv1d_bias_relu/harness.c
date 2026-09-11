#include "harness_common.h"
#include "kernel_api.h"

/* L=512 output samples, K=5 taps, C=16 channels (interleaved, channels-last).
 * x has L_in = L+K-1 = 516 samples per channel (VALID padding => no zero-pad needed).
 * Output: [L][C] = 8192 elements.
 * Not a multiple of 128 per channel (512 = 4*128, even, but K*C=80 bytes exercises
 * the multi-tap path). Edge: channel 0 bias is a large negative (tests relu clamp). */
#define L     512
#define K     5
#define C     16
#define L_IN  (L + K - 1)   /* 516 */
#define OLEN  (L * C)        /* 8192 */
#define XLEN  (L_IN * C)     /* 8256 */
#define TLEN  (K * C)        /* 80 */

static int8_t  x_buf[XLEN]  HVX_ALIGN;
static int8_t  taps_buf[TLEN] HVX_ALIGN;
static int32_t bias_buf[C]   HVX_ALIGN;
static int8_t  out_buf[OLEN] HVX_ALIGN;
static int8_t  ref_buf[OLEN] HVX_ALIGN;

/* Scalar reference for a single output element */
static int8_t ref_elem(int i, int c) {
    int32_t acc = 0;
    for (int k = 0; k < K; k++)
        acc += (int32_t)x_buf[(i + k) * C + c] * (int32_t)taps_buf[k * C + c];
    int32_t biased = acc + bias_buf[c];
    int32_t relu_v = biased < 0 ? 0 : biased;
    if (relu_v >  127) relu_v =  127;
    if (relu_v < -128) relu_v = -128;
    return (int8_t)relu_v;
}

int main(void) {
    uint32_t s = 0xBEEF13u;
    for (int i = 0; i < XLEN;  i++) x_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < TLEN;  j++) taps_buf[j]  = (int8_t)(hvx_lcg(&s) >> 24);
    for (int c = 0; c < C;     c++) bias_buf[c]  = (int32_t)((hvx_lcg(&s) & 0xFFFF) - 0x8000);

    /* Edge cases: large negative bias on ch0 triggers relu clamp */
    bias_buf[0]  = -200000;
    /* Large positive bias on ch1 triggers saturation to +127 */
    bias_buf[1]  =  200000;
    /* Max-magnitude products */
    x_buf[0]    = -128; taps_buf[0] = -128;
    x_buf[C]    =  127; taps_buf[C] = -128;

    /* Build reference */
    for (int i = 0; i < L; i++)
        for (int c = 0; c < C; c++)
            ref_buf[i * C + c] = ref_elem(i, c);

    /* Poison output */
    for (int i = 0; i < OLEN; i++) out_buf[i] = (int8_t)0xA5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, taps_buf, bias_buf, out_buf, L, K, C); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < OLEN; i++) {
        if (out_buf[i] != ref_buf[i]) {
            errors++;
            if (fb < 0) { fb = i; gotv = (long)(int8_t)out_buf[i]; expv = (long)(int8_t)ref_buf[i]; }
        }
    }
    hvx_report(errors, OLEN, fb, gotv, expv);
    return errors ? 1 : 0;
}
