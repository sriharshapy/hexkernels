#include "harness_common.h"
#include "kernel_api.h"

static uint8_t  Q[GQA_H_Q  * GQA_S * GQA_D] HVX_ALIGN;
static int8_t   K[GQA_H_KV * GQA_S * GQA_D] HVX_ALIGN;
static uint16_t out[GQA_H_Q * GQA_S * GQA_S] HVX_ALIGN;
static uint16_t ref[GQA_H_Q * GQA_S * GQA_S] HVX_ALIGN;

int main(void) {
    uint32_t s = 0xE51Au;
    /* Q in 0..7 (positive: int8==uint8), K in -3..3. Worst-case |acc| =
     * D*7*3 = 1344 -> |acc*17/16| = 1428 < 2048, so the 12-bit requant field
     * is exact (no HMX saturation) and bit-exactness holds. */
    /* NOTE: an LCG's LOW bits have a much shorter period than the full 32-bit
     * state (Hull-Dobell) -- taking `% 8` of the raw low bits repeats with a
     * period that evenly divides 4096 = GQA_S*GQA_D, which would silently
     * make head0's and head1's Q identical and mask a wrong-head-offset bug.
     * Draw from higher, better-mixed bits instead. */
    for (int i = 0; i < GQA_H_Q  * GQA_S * GQA_D; i++) Q[i] = (uint8_t)((hvx_lcg(&s) >> 13) % 8);
    for (int i = 0; i < GQA_H_KV * GQA_S * GQA_D; i++) K[i] = (int8_t)((int)((hvx_lcg(&s) >> 13) % 7) - 3);

    /* scalar reference: per query head, shared KV head h_kv = h_q / GROUP_SIZE */
    for (int hq = 0; hq < GQA_H_Q; hq++) {
        int hkv = hq / GQA_GROUP_SIZE;
        const uint8_t *Qh = Q + (size_t)hq  * GQA_S * GQA_D;
        const int8_t  *Kh = K + (size_t)hkv * GQA_S * GQA_D;
        uint16_t      *Rh = ref + (size_t)hq * GQA_S * GQA_S;
        for (int i = 0; i < GQA_S; i++)
            for (int j = 0; j < GQA_S; j++) {
                int acc = 0;
                for (int d = 0; d < GQA_D; d++) acc += (int)Qh[i*GQA_D+d] * (int)Kh[j*GQA_D+d];
                Rh[i*GQA_S+j] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }
    }

    for (int i = 0; i < GQA_H_Q * GQA_S * GQA_S; i++)
        *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < GQA_H_Q * GQA_S * GQA_S; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, GQA_H_Q * GQA_S * GQA_S, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
