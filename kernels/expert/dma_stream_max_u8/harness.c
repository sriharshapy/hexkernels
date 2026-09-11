/* dma_stream_max_u8 harness (v6, Group C: dma double-buffer streamed
 * reduction). Streamed unsigned-max reduction over a large uint8 array.
 * Harness owns main(): seeds deterministic inputs with the true max planted
 * at a random (non-obvious) position, maps VTCM identity, times the
 * candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 1100000   /* a ~1.05MB > L2; not a multiple of 16384 or 128 */
#endif

static uint8_t a[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0xA5A5F00Du;
    for (int i = 0; i < N; i++) a[i] = (uint8_t)(hvx_lcg(&s) & 0x7F); /* keep < 0xF0 */
    /* Plant the true max NOT at an edge, buried mid-stream. */
    a[N/2 + 37] = 0xFE;
    a[3] = 0x00;   /* min edge case */

    uint8_t ref = 0;
    for (int i = 0; i < N; i++) if (a[i] > ref) ref = a[i];

    uint8_t out = 0;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, &out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out != ref) ? 1 : 0;
    hvx_report(errors, 1, errors ? 0 : -1, (long)out, (long)ref);
    return errors ? 1 : 0;
}
