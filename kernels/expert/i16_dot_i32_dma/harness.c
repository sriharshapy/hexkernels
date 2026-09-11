/* i16_dot_i32_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int16 dot product at large N. Correctness contract identical
 * to i16_dot_i32 (out[0] = sum a[i]*b[i], int32 accumulator). Inputs are bounded
 * to [-16,16] so the exact sum fits int32 with no overflow. Harness owns main();
 * maps VTCM identity before the timed call; HVX init keeps simulated cycles
 * within budget. Reference is int64-accumulated then compared exactly. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 393216   /* 2 int16 inputs -> 1.5MB working set > 1MB L2 -> DDR-bound */
#endif

static int16_t a[N]   HVX_ALIGN;
static int16_t b[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;

static inline int16_t clampv(int v) { v = v % 33 - 16; return (int16_t)v; }

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill: replicate a bounded int16 pattern across all vectors. */
    {
        const int lanes16 = sizeof(HVX_Vector) / 2;
        int16_t va_init[64], vb_init[64];
        for (int j = 0; j < 64; j++) { va_init[j] = clampv(j*7+3); vb_init[j] = clampv(j*11+5); }
        HVX_Vector cur_a = *(HVX_Vector *)va_init, cur_b = *(HVX_Vector *)vb_init;
        int i = 0;
        for (; i + lanes16 <= N; i += lanes16) {
            *(HVX_Vector *)(a + i) = cur_a;
            *(HVX_Vector *)(b + i) = cur_b;
        }
        for (; i < N; i++) { a[i] = clampv(i*7+3); b[i] = clampv(i*11+5); }
    }
    /* Max-magnitude products at head/tail exercise sign handling. */
    a[0]=16;  b[0]=16;    a[1]=16;  b[1]=-16;
    a[2]=-16; b[2]=-16;   a[3]=0;   b[3]=16;
    a[N-1]=16; b[N-1]=-16;

    int64_t ref64 = 0;
    for (int i = 0; i < N; i++) ref64 += (int64_t)a[i] * (int64_t)b[i];
    int32_t ref = (int32_t)ref64;

    out[0] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = ((int64_t)out[0] != ref64) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
