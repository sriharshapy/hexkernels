/* u8_sad_row_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound sum-of-absolute-differences at large N. Correctness contract
 * identical to u8_sad_row (out[0] = sum |a[i]-b[i]|, unsigned bytes). Harness owns
 * main(); maps VTCM identity before the timed call; HVX init keeps simulated
 * cycles within budget. Reference is an exact int32 scalar SAD. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 786432   /* 2 uint8 inputs -> 1.5MB working set > 1MB L2 -> DDR-bound */
#endif

static uint8_t a[N]   HVX_ALIGN;
static uint8_t b[N]   HVX_ALIGN;
static int32_t out[1] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of a[] and b[] (unsigned pseudo-ramps). */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t va_init[128], vb_init[128], va_step[128], vb_step[128];
        for (int j = 0; j < 128; j++) {
            va_init[j] = (uint8_t)(j*7+3);   vb_init[j] = (uint8_t)(j*11+5);
            va_step[j] = (uint8_t)(128*7);   vb_step[j] = (uint8_t)(128*11);
        }
        HVX_Vector cur_a = *(HVX_Vector *)va_init, cur_b = *(HVX_Vector *)vb_init;
        HVX_Vector step_a = *(HVX_Vector *)va_step, step_b = *(HVX_Vector *)vb_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector *)(a + i) = cur_a;
            *(HVX_Vector *)(b + i) = cur_b;
            cur_a = Q6_Vb_vadd_VbVb(cur_a, step_a);
            cur_b = Q6_Vb_vadd_VbVb(cur_b, step_b);
        }
        for (; i < N; i++) { a[i] = (uint8_t)(i*7+3); b[i] = (uint8_t)(i*11+5); }
    }
    /* Edge cases: max diff both directions, zero diff, tail. */
    a[0]=255; b[0]=0;   a[1]=0;   b[1]=255;
    a[2]=128; b[2]=128; a[N-1]=200; b[N-1]=50;

    /* Exact reference: sum of |a-b| (unsigned). */
    int32_t ref = 0;
    for (int i = 0; i < N; i++) { int d = (int)a[i] - (int)b[i]; ref += d < 0 ? -d : d; }

    out[0] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = (out[0] != ref) ? 1 : 0;
    hvx_report(errors, N, errors ? 0 : -1, (long)out[0], (long)ref);
    return errors ? 1 : 0;
}
