/* i8_relu_l2fetch harness (v5, L2 multi-mechanism: hvx + l2fetch).
 * Bandwidth-bound int8 ReLU at large N. Correctness contract identical to i8_relu.
 * Harness owns main(); HVX init/verify keep simulated cycles within budget.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 524288   /* 4096 vectors; working set 2*512KB=1MB -> DDR-bound, sim<=40s */
#endif

static int8_t x[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    /* Fast HVX fill of x[]. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (int8_t)(j*7+3); v_step[j] = (int8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(x + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) x[i] = (int8_t)(i*7+3);
    }
    /* Boundary values. */
    x[0]=0; x[1]=-1; x[2]=1; x[3]=-128; x[4]=127; x[5]=-127; x[6]=-64; x[7]=64;

    /* Reference (scalar semantics, computed with HVX). */
    {
        const int vlen = sizeof(HVX_Vector);
        HVX_Vector zero = Q6_V_vzero();
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = Q6_Vb_vmax_VbVb(*(const HVX_Vector *)(x + i), zero);
        for (; i < N; i++) ref[i] = (x[i] > 0) ? x[i] : 0;
    }
    /* Poison output. */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vp = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(x, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
