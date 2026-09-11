/* i8_global_maxpool_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 global-max-pool at large N (SIGNED max). Harness owns
 * main(); maps VTCM identity before the timed call; HVX init keeps cycles in
 * budget. Reference is an exact signed scalar max. The fill keeps values below
 * the injected unique max (127) and plants a -1 (unsigned 255) so an
 * unsigned-max near-miss is exposed. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112   /* 8704 vectors; working set 1.0625MB > 1MB L2 -> DDR-bound */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill: values in [0,99] so the injected 127 is a unique max and the
       injected -1/-128 do not exceed it. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128];
        for (int j = 0; j < 128; j++) v_init[j] = (int8_t)((j*7+3) % 100);
        HVX_Vector cur = *(HVX_Vector *)v_init;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(a + i) = cur;
        for (; i < N; i++) a[i] = (int8_t)((i*7+3) % 100);
    }
    a[7]   = -128;   /* minimum, must not be mistaken for max */
    a[100] = -1;     /* unsigned 255: defeats an unsigned-max near-miss */
    a[N-1] = 127;    /* unique signed max in the tail region */

    /* Exact reference: signed scalar max. */
    int8_t m = -128;
    for (int i = 0; i < N; i++) if (a[i] > m) m = a[i];
    int8_t ref = m;

    out[0] = (int8_t)0xA5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
