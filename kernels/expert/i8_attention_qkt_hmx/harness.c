#include "harness_common.h"
#include "kernel_api.h"
#define S 64
#define D 64

static uint8_t  Q[S*D]   HVX_ALIGN;
static int8_t   K[S*D]   HVX_ALIGN;
static uint16_t out[S*S] HVX_ALIGN;
static uint16_t ref[S*S] HVX_ALIGN;

int main(void) {
    uint32_t s = 0x71C3u;
    /* Q in 0..7 (positive: int8==uint8), K in -3..3. Worst-case |acc| = D*7*3 =
     * 1344 -> |acc*17/16| = 1428 < 2048, so the 12-bit requant field is exact
     * (no HMX saturation) and bit-exactness holds. */
    for (int i = 0; i < S*D; i++) Q[i] = (uint8_t)(hvx_lcg(&s) % 8);
    for (int i = 0; i < S*D; i++) K[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);

    /* scalar reference: scores = Q . K^T (K row j is key vector j) + HMX 0x40 requant */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Q[i*D+d] * (int)K[j*D+d];
            ref[i*S+j] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }

    for (int i = 0; i < S*S; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, out, S, D); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*S; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, S*S, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
