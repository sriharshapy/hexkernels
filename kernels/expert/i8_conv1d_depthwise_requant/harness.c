#include "harness_common.h"
#include "kernel_api.h"

/* C=16 channels, L=256 output samples/channel, K=7 taps/channel.
 * L_in per channel = L+K-1 = 262. Total x = C*L_in = 4192.
 * Total out = C*L = 4096.
 * Sweep 3 (mult,shift,zp) sets to prevent hardcoding of quant params. */
#define C     16
#define L     128
#define K     7
#define L_IN  (L + K - 1)   /* 262 */
#define XLEN  (C * L_IN)     /* 4192 */
#define TLEN  (C * K)        /* 112 */
#define OLEN  (C * L)        /* 4096 */

static int8_t  x_buf[XLEN]   HVX_ALIGN;
static int8_t  taps_buf[TLEN] HVX_ALIGN;
static int8_t  out_buf[OLEN]  HVX_ALIGN;
static int8_t  ref_buf[OLEN]  HVX_ALIGN;

static const int32_t MULTS[]  = { 5, 1 };
static const int     SHIFTS[] = { 6, 0 };
static const int8_t  ZPS[]    = { 0, -5 };
#define NSETS ((int)(sizeof(MULTS)/sizeof(MULTS[0])))

static int8_t requant(int32_t acc, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)acc * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp;
    if (r >  127) r =  127;
    if (r < -128) r = -128;
    return (int8_t)r;
}

int main(void) {
    uint32_t s = 0xA7C3E1u;
    for (int i = 0; i < XLEN; i++) x_buf[i]    = (int8_t)(hvx_lcg(&s) >> 24);
    for (int j = 0; j < TLEN; j++) taps_buf[j]  = (int8_t)(hvx_lcg(&s) >> 24);

    /* Max-magnitude products */
    x_buf[0]      = -128; taps_buf[0]  = -128;
    x_buf[L_IN]   = -128; taps_buf[K]  = -128;
    x_buf[XLEN-1] = -128;

    int total_errors = 0, fb = -1; long gotv = 0, expv = 0;

    for (int p = 0; p < NSETS; p++) {
        int32_t mult  = MULTS[p];
        int     shift = SHIFTS[p];
        int8_t  zp    = ZPS[p];

        /* Build reference */
        for (int c = 0; c < C; c++) {
            const int8_t *xc   = x_buf    + c * L_IN;
            const int8_t *tapc = taps_buf + c * K;
            int8_t       *rc   = ref_buf  + c * L;
            for (int i = 0; i < L; i++) {
                int32_t acc = 0;
                for (int k = 0; k < K; k++)
                    acc += (int32_t)xc[i + k] * (int32_t)tapc[k];
                rc[i] = requant(acc, mult, shift, zp);
            }
        }

        /* Poison */
        for (int i = 0; i < OLEN; i++) out_buf[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x_buf, taps_buf, out_buf, C, L, K, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < OLEN; i++) {
            if (out_buf[i] != ref_buf[i]) {
                total_errors++;
                if (fb < 0) { fb = p * OLEN + i; gotv = (long)(int8_t)out_buf[i]; expv = (long)(int8_t)ref_buf[i]; }
            }
        }
    }
    hvx_report(total_errors, OLEN * NSETS, fb, gotv, expv);
    return total_errors ? 1 : 0;
}
