/* u8_count_ge_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound "count elements >= t" reduction at large N. Correctness
 * contract identical to u8_count_ge (inclusive >=). The threshold t is a fixed
 * runtime constant (T=100) with exact-boundary a[i]==t values injected so a
 * strict-'>' candidate is wrong. Harness owns main(); maps VTCM identity before
 * the timed call; HVX init keeps simulated cycles within budget. Reference is an
 * exact scalar count cross-checked against the injected boundary structure.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112   /* 8704 vectors; working set 1.0625MB > 1MB L2 -> DDR-bound */
#endif
#define T 100       /* fixed runtime threshold (passed as an argument) */

static uint8_t a[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[] (unsigned bytes, pseudo-ramp). */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (uint8_t)(j*7+3); v_step[j] = (uint8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (uint8_t)(i*7+3);
    }
    /* Inject exact boundary hits (a==T) and extremes so a strict '>' is wrong. */
    a[0] = T; a[1] = T; a[2] = T; a[3] = T;          /* four a==t (>= counts, > does not) */
    a[4] = T-1; a[5] = 0; a[6] = 255; a[N-1] = T;    /* below, extremes, tail boundary */

    /* Exact reference: inclusive count a[i] >= T. */
    int32_t ref = 0;
    for (int i = 0; i < N; i++) if (a[i] >= (uint8_t)T) ref++;

    out[0] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, (uint8_t)T, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
