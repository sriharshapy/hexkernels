/* vtcm_stage_matmul_i8 harness (v6, Group C: hvx + dma + vtcm).
 * GEMM with a LARGE streamed operand A [M x K] and a SMALL staged operand X
 * [K x N] (N<=4, reused unchanged across every row of A). Harness owns
 * main(): maps VTCM identity, seeds deterministic inputs (plus edge-value
 * injections), computes the int64-checked scalar reference directly,
 * poisons C, times the candidate (kernel-only pcycles), bit-exact compares.
 *
 * Size macros are named HM/HK/HN (NOT M/K/N) so a fast-debug override like
 * -DHK=500 on the compiler command line cannot collide with the candidate's
 * own M/K/N parameter identifiers (those files are compiled in the same
 * invocation as this harness).
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef HM
#define HM 161
#endif
#ifndef HK
#define HK 8101   /* NOT a multiple of 4 -- vrmpy k-group tail */
#endif
#ifndef HN
#define HN 4
#endif

static int8_t  A[(size_t)HM * HK]  HVX_ALIGN;
static int8_t  X[(size_t)HK * HN]  HVX_ALIGN;
static int32_t C[(size_t)HM * HN]  HVX_ALIGN;
static int32_t ref[(size_t)HM * HN] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x51AB99CCu;
    for (size_t i = 0; i < (size_t)HM * HK; i++) A[i] = (int8_t)(hvx_lcg(&s) >> 24);
    for (size_t i = 0; i < (size_t)HK * HN; i++) X[i] = (int8_t)(hvx_lcg(&s) >> 24);

    /* Edge-value injections (sign + max-magnitude + tail-k coverage). */
    A[0] = 127;                       X[0 * HN + 0] = 127;   /* max +contribution, k=0 */
    A[1] = -128;                      X[1 * HN + 0] = 127;   /* sign in the reduction, k=1 */
    A[HK - 1] = 127;                  X[(size_t)(HK - 1) * HN + (HN - 1)] = 127; /* last (tail) k, last col */
    A[HK - 2] = -128;                 X[(size_t)(HK - 2) * HN + (HN - 1)] = -128; /* tail-group sign */

    for (int m = 0; m < HM; m++)
        for (int j = 0; j < HN; j++) {
            int64_t acc = 0;
            for (int k = 0; k < HK; k++)
                acc += (int64_t)A[(size_t)m * HK + k] * (int64_t)X[(size_t)k * HN + j];
            ref[(size_t)m * HN + j] = (int32_t)acc;   /* fits int32 at these bounds */
        }

    for (size_t i = 0; i < (size_t)HM * HN; i++) C[i] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(A, X, C, HM, HK, HN); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0; long fb = -1, gotv = 0, expv = 0;
    for (size_t i = 0; i < (size_t)HM * HN; i++) {
        if (C[i] != ref[i]) {
            errors++;
            if (fb < 0) { fb = (long)i; gotv = (long)C[i]; expv = (long)ref[i]; }
        }
    }
    hvx_report(errors, HM * HN, (int)fb, gotv, expv);
    return errors ? 1 : 0;
}
