/* i8_gather_vtcm harness (v6, group H holdout: dma + vtcm + hvx).
 * Embedding-style D-wide row gather (out[i,:]=table[idx[i],:]) with a table
 * larger than L1 (64KB) but small enough to VTCM-stage, and a large T so
 * total output exceeds 1MB (>> L2) -- a random-row-access / bandwidth-bound
 * gather. Harness owns main(): maps VTCM identity, fills the table + random
 * indices (deterministic LCG) with edge cases injected, builds the scalar
 * reference gather, poisons out, times the candidate (kernel-only pcycles),
 * bit-exact compares. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef VOCAB
#define VOCAB 512            /* table = 512*128 = 64KB: > L1, VTCM-stageable */
#endif
#define E     128            /* embedding dim, one HVX vector */
#ifndef T
#define T 8199               /* tokens; out = 8199*128 ~= 1.05MB > L2 */
#endif

static int8_t  table[(size_t)VOCAB * E] HVX_ALIGN;
static int32_t idx[T]                    HVX_ALIGN;
static int8_t  out[(size_t)T * E]        HVX_ALIGN;
static int8_t  ref[(size_t)T * E]        HVX_ALIGN;

static void gather_ref(const int8_t *tab, const int32_t *id, int8_t *r, int t, int e) {
    for (int i = 0; i < t; i++) {
        const int8_t *src = tab + (int64_t)id[i] * e;
        int8_t *dst = r + (int64_t)i * e;
        for (int k = 0; k < e; k++) dst[k] = src[k];
    }
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xF00DFACEu;
    for (size_t i = 0; i < (size_t)VOCAB * E; i++) table[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (int i = 0; i < T; i++) idx[i] = (int32_t)(hvx_lcg(&s) % VOCAB);

    /* Edge cases: first/last vocab row, repeated index, boundary token positions. */
    idx[0] = 0; idx[1] = VOCAB - 1; idx[2] = 0; idx[3] = VOCAB / 2;
    idx[T - 1] = VOCAB - 1; idx[T - 2] = 0;

    gather_ref(table, idx, ref, T, E);

    for (size_t i = 0; i < (size_t)T * E; i++) out[i] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(table, idx, out, T, E, VOCAB); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    long errors = 0, fb = -1, gotv = 0, expv = 0;
    for (size_t i = 0; i < (size_t)T * E; i++) {
        if (out[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = (long)i; gotv = (long)out[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report((int)errors, T * E, (int)fb, gotv, expv);
    return errors ? 1 : 0;
}
