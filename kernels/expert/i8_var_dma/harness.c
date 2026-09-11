/* i8_var_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 integer-variance reduction at large N. Correctness
 * contract identical to i8_var_i32:
 *   out[0] = (n*sum(a^2) - (sum a)^2) / n   (integer floor, int64 intermediates).
 * Harness owns main(); maps VTCM identity before the timed call; HVX init keeps
 * cycles in budget. Reference computed exactly in int64. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112   /* 8704 vectors; working set 1.0625MB > 1MB L2 -> DDR-bound */
#endif

static int8_t  a[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*7+3); v_step[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (int8_t)(i*7+3);
    }
    /* Max-magnitude values stress the int64 sum-of-squares path. */
    a[0] = 127; a[1] = -128; a[N-2] = 127; a[N-1] = -128;

    /* Exact reference in int64. */
    int64_t sumx = 0, sumsq = 0;
    for (int i = 0; i < N; i++) { int64_t v = (int64_t)a[i]; sumx += v; sumsq += v*v; }
    int32_t ref = (int32_t)(((int64_t)N * sumsq - sumx * sumx) / (int64_t)N);

    out[0] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
