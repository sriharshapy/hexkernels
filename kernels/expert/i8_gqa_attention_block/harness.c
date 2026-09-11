/* i8_gqa_attention_block harness (v5, L3 -- grouped-query attention BLOCK).
 * The harness owns main(): fills seeded inputs, computes an INDEPENDENT scalar
 * golden reference for the whole GQA block (QK^T -> scale+softmax -> A.V, with
 * H_Q query heads sharing H_KV key/value heads), builds the runtime exp LUT,
 * poisons the output, times the candidate (kernel-only pcycles) and does a
 * bit-exact compare. Enables the HMX context AND installs an identity VTCM
 * translation so a candidate can compose HMX + VTCM + HVX. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <stdint.h>

static uint8_t  Q[GQA_H_Q  * GQA_S * GQA_D] HVX_ALIGN;
static int8_t   K[GQA_H_KV * GQA_S * GQA_D] HVX_ALIGN;
static int8_t   V[GQA_H_KV * GQA_S * GQA_D] HVX_ALIGN;
static uint16_t out[GQA_H_Q * GQA_S * GQA_D] HVX_ALIGN;
static uint16_t ref[GQA_H_Q * GQA_S * GQA_D] HVX_ALIGN;
static uint8_t  exp_lut[256] HVX_ALIGN;

/* sign-extend a 12-bit two's-complement requant field */
static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}
static inline int clamp_i8(int v) {
    if (v >  127) return  127;
    if (v < -128) return -128;
    return v;
}

/* INDEPENDENT scalar golden: full GQA block computed straight from the math
 * (NOT reusing baseline.c) so a buggy shared reference can't hide. */
static void gqa_block_ref(const uint8_t *q, const int8_t *k, const int8_t *v,
                          uint16_t *r, const uint8_t *lut) {
    static int8_t  scaled[GQA_S*GQA_S];
    static uint8_t probs[GQA_S*GQA_S];
    for (int hq = 0; hq < GQA_H_Q; hq++) {
        int hkv = hq / GQA_GROUP_SIZE;
        const uint8_t *Qh = q + (size_t)hq  * GQA_S * GQA_D;
        const int8_t  *Kh = k + (size_t)hkv * GQA_S * GQA_D;
        const int8_t  *Vh = v + (size_t)hkv * GQA_S * GQA_D;
        uint16_t      *Rh = r + (size_t)hq  * GQA_S * GQA_D;

        /* Stage 1: scores = Q.K^T -> 0x40 requant -> scale + clamp to int8 */
        for (int i = 0; i < GQA_S; i++)
            for (int j = 0; j < GQA_S; j++) {
                int acc = 0;
                for (int d = 0; d < GQA_D; d++) acc += (int)Qh[i*GQA_D+d] * (int)Kh[j*GQA_D+d];
                int s12 = hvx_hmx_requant_0x40(acc);
                scaled[i*GQA_S+j] = (int8_t)clamp_i8(sx12(s12) >> 4);
            }
        /* Stage 2: row-wise softmax over the KEY axis j (fixed-point LUT) */
        for (int i = 0; i < GQA_S; i++) {
            int m = scaled[i*GQA_S+0];
            for (int j = 1; j < GQA_S; j++) if (scaled[i*GQA_S+j] > m) m = scaled[i*GQA_S+j];
            int Sr = 0;
            for (int j = 0; j < GQA_S; j++) {
                int diff = (int)scaled[i*GQA_S+j] - m;
                if (diff < -255) diff = -255;
                uint8_t e = lut[diff + 255];
                probs[i*GQA_S+j] = e;
                Sr += (int)e;
            }
            int half = Sr / 2;
            for (int j = 0; j < GQA_S; j++) {
                int ej = (int)probs[i*GQA_S+j];
                probs[i*GQA_S+j] = (uint8_t)((ej * 255 + half) / Sr);
            }
        }
        /* Stage 3: out = probs . V -> 0x40 requant */
        for (int i = 0; i < GQA_S; i++)
            for (int d = 0; d < GQA_D; d++) {
                int acc = 0;
                for (int j = 0; j < GQA_S; j++) acc += (int)probs[i*GQA_S+j] * (int)Vh[j*GQA_D+d];
                Rh[i*GQA_D+d] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }
    }
}

int main(void) {
    /* Map VTCM identity so a candidate can stage crouton tiles on-chip. */
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x9C4B21E7u;
    /* Q in 0..7 (positive: int8==uint8), K,V in -3..3.  QK^T: |acc|<=D*7*3=1344 ->
     * *17/16=1428<2048. A.V: probs is a normalized row (sum ~255), V<=3 ->
     * |acc|<=~287*3=861 -> *17/16=915<2048.  Whole integer block BIT-EXACT.
     * NOTE: draw from higher, better-mixed LCG bits (>>13) -- the raw low bits
     * have a short period that evenly divides S*D=4096 and would silently make
     * head0's and head1's Q identical, masking a wrong-head-grouping bug. */
    for (int i = 0; i < GQA_H_Q  * GQA_S * GQA_D; i++) Q[i] = (uint8_t)((hvx_lcg(&s) >> 13) % 8);
    for (int i = 0; i < GQA_H_KV * GQA_S * GQA_D; i++) K[i] = (int8_t)((int)((hvx_lcg(&s) >> 13) % 7) - 3);
    for (int i = 0; i < GQA_H_KV * GQA_S * GQA_D; i++) V[i] = (int8_t)((int)((hvx_lcg(&s) >> 13) % 7) - 3);

    /* Edge case: query head 1, row 0 all-zeros -> all scores 0 -> uniform softmax. */
    for (int d = 0; d < GQA_D; d++) Q[(1*GQA_S + 0)*GQA_D + d] = 0;

    /* Fixed-point exp LUT (Q16 decay recurrence, no libm): exp_lut[i] =
     * round(255 * exp((i-255)/24)); index 255 = exp(0) = 255. */
    {
        const uint32_t DECAY_Q16 = 62865u;   /* round(exp(-1/24) * 65536) */
        uint64_t vq = (uint64_t)255u << 16;
        for (int i = 255; i >= 0; i--) {
            exp_lut[i] = (uint8_t)((vq + 32768u) >> 16);
            vq = (vq * DECAY_Q16) >> 16;
        }
        exp_lut[255] = 255;
    }

    gqa_block_ref(Q, K, V, ref, exp_lut);

    for (int i = 0; i < GQA_H_Q * GQA_S * GQA_D; i++)
        *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, V, out, exp_lut); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < GQA_H_Q * GQA_S * GQA_D; i++) {
        unsigned short g = out[i], e = ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; } }
    }
    hvx_report_u16(errors, GQA_H_Q * GQA_S * GQA_D, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
