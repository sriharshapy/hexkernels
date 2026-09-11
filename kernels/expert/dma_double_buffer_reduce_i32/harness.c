/* dma_double_buffer_reduce_i32 harness (v6, Group C: dma + vtcm double-buffer).
 * Sum-reduction of a bounded int32 array at large N (out = int64 exact sum).
 * Harness owns main(): seeds deterministic bounded inputs (|a[i]|<=1000) with
 * a few boundary +-1000 values, computes the exact int64 reference sum,
 * poisons out, maps VTCM identity, times the candidate (kernel-only
 * pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 300000   /* 1.2MB int32 > L2; 73 full 16KB(=4096-elem) tiles + 992 remainder */
#endif

static int32_t a[N]   HVX_ALIGN;
static int64_t out[1] HVX_ALIGN;

/* Map an LCG draw into the bounded range [-1000, 1000]. */
static inline int32_t clampv(uint32_t r) { return (int32_t)(r % 2001u) - 1000; }

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xC0FFEE42u;
    for (int i = 0; i < N; i++) a[i] = clampv(hvx_lcg(&s));
    /* Boundary +-1000 values, incl. at the head/tail and near the tail split. */
    a[0] = 1000; a[1] = -1000; a[2] = 1000; a[3] = -1000;
    a[N-2] = 1000; a[N-1] = -1000;
    if (N > 299008) { a[299007] = 1000; a[299008] = -1000; }   /* around the 4096-elem tile boundary */

    int64_t ref = 0;
    for (int i = 0; i < N; i++) ref += (int64_t)a[i];

    out[0] = (int64_t)0xA5A5A5A5A5A5A5A5LL;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
