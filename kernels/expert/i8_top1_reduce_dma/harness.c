/* i8_top1_reduce_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8 top-1 reduction at large N. Correctness contract identical
 * to i8_top1_reduce: out[0] = max value, out[1] = first index of that value.
 * Harness owns main(); maps identity VTCM before the timed call; HVX-fills a[]
 * bounded to [-120,120] so the injected 127 (first at 250, tie at 700250) is the
 * unique maximum. Reference is an exact scalar max/argmax (strict >).
 * N=1114112 is a multiple of 128 (no byte tail).
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112
#endif

static int8_t  a[N]   HVX_ALIGN;
static int32_t out[2] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*7+3); v_step[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        HVX_Vector vhi = Q6_V_vsplat_R(0x78787878);   /* +120 */
        HVX_Vector vlo = Q6_V_vsplat_R(0x88888888);   /* -120 */
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            HVX_Vector c = Q6_Vb_vmin_VbVb(cur, vhi);
            c = Q6_Vb_vmax_VbVb(c, vlo);
            *(HVX_Vector *)(a + i) = c;
            cur = Q6_Vb_vadd_VbVb(cur, step);
        }
        for (; i < N; i++) { int v = (int8_t)(i*7+3); if (v>120) v=120; if (v<-120) v=-120; a[i]=(int8_t)v; }
    }
    a[250] = 127; a[700250] = 127;   /* first at 250, tie later */

    int bv = (int)a[0], bi = 0;
    for (int i = 1; i < N; i++) if ((int)a[i] > bv) { bv = a[i]; bi = i; }
    int32_t ref_val = bv;   /* 127 */
    int32_t ref_idx = bi;   /* 250 */

    out[0] = (int32_t)0xA5A5A5A5;
    out[1] = (int32_t)0xA5A5A5A5;

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, N, out); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, first_bad = -1; long gotv = 0, expv = 0;
    if (out[0] != ref_val) { errors++; if (first_bad < 0) { first_bad = 0; gotv = out[0]; expv = ref_val; } }
    if (out[1] != ref_idx) { errors++; if (first_bad < 0) { first_bad = 1; gotv = out[1]; expv = ref_idx; } }
    hvx_report(errors, N, first_bad, gotv, expv);
    return errors ? 1 : 0;
}
