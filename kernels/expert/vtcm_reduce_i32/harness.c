/* vtcm_reduce_i32 harness (v6, Group C: dma + vtcm reduction).
 * int32 sum-reduction at large N. Harness owns main(): seeds deterministic
 * input, computes the exact scalar sum reference, poisons out, maps VTCM
 * identity, times the candidate (kernel-only pcycles), exact-compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 284000   /* 1.136MB > L2; 69 full 16KB tiles + 1376-elem remainder */
#endif

static int32_t a[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x2468ACE1u;
    for (int i = 0; i < N; i++) a[i] = (int32_t)hvx_lcg(&s);
    /* Boundary values (32-bit wraparound edge cases). */
    a[0] = 2147483647;  a[1] = -2147483647 - 1;
    a[N-2] = 2147483647; a[N-1] = 1;

    int32_t ref = 0;
    for (int i = 0; i < N; i++) ref += a[i];   /* wraparound sum, order-independent */

    out[0] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
