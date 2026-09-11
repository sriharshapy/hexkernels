#include "harness_common.h"
#include "kernel_api.h"
#define S 32
#define D 32

static uint8_t Q[S*D]   HVX_ALIGN;
static int8_t  K[S*D]   HVX_ALIGN;
static int8_t  V[S*D]   HVX_ALIGN;
static int32_t out[S*D] HVX_ALIGN;
static int32_t ref[S*D] HVX_ALIGN;

static inline int sx12(int f) { int v = f & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }

int main(void) {
    uint32_t s = 0xA77Fu;
    /* Q,K in {0,1}: scores in [0,32] -> Sc = sx12(round(scores*17/16)) in [0,34]
     * (non-negative, safe to re-pack as a uint8 HMX activation). V in {-1,0,1}:
     * |acc2| <= 32*34*1 = 1088 -> |out| <= sx12(round(1088*17/16)) = 1157 < 2048. */
    for (int i = 0; i < S*D; i++) Q[i] = (uint8_t)(hvx_lcg(&s) % 2);
    for (int i = 0; i < S*D; i++) K[i] = (int8_t)(hvx_lcg(&s) % 2);
    for (int i = 0; i < S*D; i++) V[i] = (int8_t)((int)(hvx_lcg(&s) % 3) - 1);

    static int Sc[S*S];
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Q[i*D+d] * (int)K[j*D+d];
            Sc[i*S + j] = sx12((acc * 17 + 8) >> 4);
        }
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += Sc[i*S + j] * (int)V[j*D + d];
            ref[i*D + d] = sx12((acc * 17 + 8) >> 4);
        }

    for (int i = 0; i < S*D; i++) *((volatile int32_t *)&out[i]) = 0x5A5A5A5A; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, V, out, S, D); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gotv = out[i]; expv = ref[i]; } }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
