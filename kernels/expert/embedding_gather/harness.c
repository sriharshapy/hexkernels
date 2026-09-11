#include "harness_common.h"
#include "kernel_api.h"
#include <stdint.h>

#define VOCAB_SIZE 64
#define E          128   /* embedding dim, multiple of 128 for HVX */
#define T          33    /* token count, NOT a multiple of 128 */

static int8_t  table[VOCAB_SIZE * E] HVX_ALIGN;
static int32_t idx[T]                HVX_ALIGN;
static int8_t  out[T * E]            HVX_ALIGN;
static int8_t  ref[T * E]            HVX_ALIGN;

/* Reference: simple gather */
static void gather_ref(const int8_t *tab, const int32_t *id, int8_t *r,
                       int t, int e) {
    for (int i = 0; i < t; i++) {
        const int8_t *src = tab + (int)id[i] * e;
        int8_t *dst = r + i * e;
        for (int k = 0; k < e; k++) dst[k] = src[k];
    }
}

/*
 * Two index sweeps with distinct tables (anti-hardcode).
 * The candidate MUST use the runtime table and idx pointers.
 */
#define NSETS 2

int main(void) {
    uint32_t s = 0x12345678u;

    int8_t  tables[NSETS][VOCAB_SIZE * E];
    int32_t idxs  [NSETS][T];

    for (int k = 0; k < NSETS; k++) {
        for (int i = 0; i < VOCAB_SIZE * E; i++)
            tables[k][i] = (int8_t)(hvx_lcg(&s) >> 24);
        for (int i = 0; i < T; i++)
            idxs[k][i] = (int32_t)((hvx_lcg(&s) >> 24) % VOCAB_SIZE);
    }

    /* Edge cases */
    idxs[0][0]  = 0;             /* first vocab row */
    idxs[0][1]  = VOCAB_SIZE-1;  /* last vocab row */
    idxs[0][2]  = 0;             /* repeated index */
    idxs[0][T-1] = VOCAB_SIZE/2; /* last token position */
    idxs[1][0]  = VOCAB_SIZE-1;
    idxs[1][T-1] = 0;

    int errors = 0, fb = -1;
    long gotv = 0, expv = 0;

    for (int k = 0; k < NSETS; k++) {
        /* Copy table and idx into aligned buffers */
        for (int i = 0; i < VOCAB_SIZE * E; i++) table[i] = tables[k][i];
        for (int i = 0; i < T; i++) idx[i] = idxs[k][i];

        gather_ref(table, idx, ref, T, E);

        for (int i = 0; i < T * E; i++) out[i] = (int8_t)0xA5;

        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx, out, T, E); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

        for (int i = 0; i < T * E; i++) {
            if (out[i] != ref[i]) {
                errors++;
                if (fb < 0) {
                    fb = k * T * E + i;
                    gotv = (long)out[i];
                    expv = (long)ref[i];
                }
            }
        }
    }

    hvx_report(errors, T * E * NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
