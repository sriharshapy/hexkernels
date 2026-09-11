/* l2fetch_rolling_add_i8 harness (v6, Group C: rolling l2fetch + unroll).
 * Wraparound int8 add at large N (>1MB per stream). Harness owns main():
 * seeds deterministic inputs with wrap-boundary values, poisons out, times
 * the candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 600064   /* 4688 vectors; a+b+out ~1.72MB total > L2 */
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t b[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t va_init[128], vb_init[128], va_step[128], vb_step[128];
        for (int j = 0; j < 128; j++) {
            va_init[j] = (int8_t)(j * 7 + 3);
            vb_init[j] = (int8_t)(j * 11 + 5);
            va_step[j] = (int8_t)(128 * 7);
            vb_step[j] = (int8_t)(128 * 11);
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
        for (; i < N; i++) { a[i] = (int8_t)(i*7+3); b[i] = (int8_t)(i*11+5); }
    }
    a[0]=127; b[0]=1; a[1]=-128; b[1]=-1; a[2]=127; b[2]=127; a[3]=-128; b[3]=-128;

    for (int i = 0; i < N; i++) ref[i] = (int8_t)(a[i] + b[i]);

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
