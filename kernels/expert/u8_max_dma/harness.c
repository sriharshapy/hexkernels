/* u8_max_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound uint8 max-reduction at large N. Correctness contract identical
 * to u8_max_reduce. Harness owns main(); maps VTCM identity before the timed
 * call; HVX init keeps cycles in budget. Reference is an exact scalar max. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112   /* 8704 vectors; working set 1.0625MB > 1MB L2 -> DDR-bound */
#endif

static uint8_t a[N]   HVX_ALIGN;
static uint8_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[] with a deterministic byte pattern kept below 250 so the
       injected 254 is a unique maximum. */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (uint8_t)((j*7+3) % 250); v_step[j] = 0; }
        HVX_Vector cur = *(HVX_Vector *)v_init;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(a + i) = cur;
        for (; i < N; i++) a[i] = (uint8_t)((i*7+3) % 250);
    }
    a[N-1] = 254;   /* unique max in the tail region */

    /* Exact reference. */
    uint8_t m = 0;
    for (int i = 0; i < N; i++) if (a[i] > m) m = a[i];
    uint8_t ref = m;

    out[0] = 0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
