/* dma_scatter_vtcm_i8 harness (v6, Group C: scatter + vtcm).
 * out[idx[i]] = (int32_t)values[i] over a large DDR-resident int32 table,
 * idx a permutation of [0,N). Harness owns main(): fills values
 * (deterministic LCG, int8 range) and an affine permutation index array,
 * builds the scalar reference, poisons out, maps VTCM identity so a
 * scatter candidate can stage the table on-chip, times the candidate
 * (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 32768   /* table words = 128KB (VTCM-resident, exceeds L1) */
#endif

static int8_t  values[N] HVX_ALIGN;
static int32_t idx[N]    HVX_ALIGN;
static int32_t out[N]    HVX_ALIGN;
static int32_t ref[N]    HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xDEECE66Du;
    for (int i = 0; i < N; i++) values[i] = (int8_t)hvx_lcg(&s);
    /* Affine permutation of [0,N): odd multiplier mod a power of two is a
     * bijection, so every destination position is written exactly once. */
    for (int i = 0; i < N; i++) idx[i] = (int32_t)(((uint32_t)i * 16385u + 7u) & (uint32_t)(N - 1));
    /* Edge cases: first/last source element, boundary destination slots. */
    values[0] = 127; values[1] = -128; values[N-1] = -1;

    for (int i = 0; i < N; i++) ref[idx[i]] = (int32_t)values[i];

    for (int i = 0; i < N; i++) out[i] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(values, idx, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gv = 0, ev = 0;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) { fb = i; gv = out[i]; ev = ref[i]; } }
    hvx_report(errors, N, fb, gv, ev);
    return errors ? 1 : 0;
}
