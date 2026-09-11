/* fp16_flash_attention_tile harness (L3 composite, holdout group H).
 * The harness owns main(): seeds deterministic Q/K/V, computes the STANDARD
 * (non-flash, full-softmax) float32 attention reference for the whole Q tile
 * (mathematically identical to the flash recurrence the candidate must
 * implement), poisons the output, times the candidate, and does a
 * tolerance compare (hvx_close_f16bits -- HVX float is non-IEEE qf16 and
 * float ops reorder, so this is NOT bit-exact). */
#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>

#define SQ 16
#define SK 64
#define DH 64

static hvx_hf Q[SQ*DH] HVX_ALIGN;
static hvx_hf K[SK*DH] HVX_ALIGN;
static hvx_hf V[SK*DH] HVX_ALIGN;
static hvx_hf O[SQ*DH] HVX_ALIGN;
static hvx_hf ref[SQ*DH] HVX_ALIGN;

static float gen_hf(uint32_t *s) {
    int q = (int)(hvx_lcg(s) % 9) - 4;   /* -4..4 */
    return (float)(q * 0.25f);            /* -1.0..1.0, exact in fp16 */
}

static void attn_ref(void) {
    for (int i = 0; i < SQ; i++) {
        float sc[SK];
        float rowmax = -1e30f;
        for (int j = 0; j < SK; j++) {
            float acc = 0.0f;
            for (int d = 0; d < DH; d++) acc += (float)Q[i*DH+d] * (float)K[j*DH+d];
            sc[j] = acc * FLASH_SCALE;
            if (sc[j] > rowmax) rowmax = sc[j];
        }
        float e[SK], rowsum = 0.0f;
        for (int j = 0; j < SK; j++) { e[j] = expf(sc[j] - rowmax); rowsum += e[j]; }
        for (int d = 0; d < DH; d++) {
            float acc = 0.0f;
            for (int j = 0; j < SK; j++) acc += e[j] * (float)V[j*DH+d];
            ref[i*DH+d] = (hvx_hf)(acc / rowsum);
        }
    }
}

int main(void) {
    uint32_t s = 0xF1A54A11u;
    for (int i = 0; i < SQ*DH; i++) Q[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < SK*DH; i++) K[i] = (hvx_hf)gen_hf(&s);
    for (int i = 0; i < SK*DH; i++) V[i] = (hvx_hf)gen_hf(&s);

    /* Edge case: row 0's query aligns strongly with the LAST key (last tile,
     * FLASH_TILE_K=16 boundary at j=48..63) -- exercises the running-max
     * getting updated (and the accumulator/sum rescaled) on the FINAL tile,
     * the case a missing/short-circuited rescale is most likely to break. */
    for (int d = 0; d < DH; d++) Q[0*DH+d] = (hvx_hf)1.0f;
    for (int d = 0; d < DH; d++) K[(SK-1)*DH+d] = (hvx_hf)1.0f;

    attn_ref();

    for (int i = 0; i < SQ*DH; i++) *((volatile unsigned short *)&O[i]) = 0xA5A5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(Q, K, V, O, SQ, SK, DH); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < SQ*DH; i++) {
        unsigned short g = *(unsigned short *)&O[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, SQ*DH, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
