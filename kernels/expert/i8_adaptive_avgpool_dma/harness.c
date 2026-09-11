/* i8_adaptive_avgpool_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound 1D adaptive average pool at large N: reduce N uint8 -> M int8,
 * window k = N/M. N=1114112, M=68 => k=16384 (each window is exactly one 16KB DMA
 * tile). Correctness contract inherits i8_adaptive_avgpool (uint8 in, int8 out,
 * truncated mean cast to int8). Harness owns main(); maps VTCM identity before
 * the timed call; HVX init keeps cycles in budget. Reference is an exact int32
 * per-window sum then /k cast to int8. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 1114112
#endif
#define M 68
#define K (N/M)   /* 16384 */

static uint8_t in[N]  HVX_ALIGN;
static int8_t  out[M] HVX_ALIGN;
static int8_t  ref[M] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fast HVX fill of in[] with a deterministic full-range byte pattern. */
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t v_init[128], v_step[128];
        for (int j = 0; j < 128; j++) { v_init[j] = (uint8_t)(j*7+3); v_step[j] = (uint8_t)(128*7); }
        HVX_Vector cur = *(HVX_Vector *)v_init, step = *(HVX_Vector *)v_step;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(in + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) in[i] = (uint8_t)(i*7+3);
    }
    /* First window all-255 -> avg 255 -> (int8)255 = -1. */
    for (int j = 0; j < K; j++) in[j] = 255;
    /* Last window all-0 -> avg 0 -> 0. */
    for (int j = 0; j < K; j++) in[(M-1)*K + j] = 0;

    /* Exact reference: per-window int32 sum then truncated mean cast to int8. */
    for (int i = 0; i < M; i++) {
        int32_t sum = 0;
        for (int j = 0; j < K; j++) sum += (int32_t)in[i*K + j];
        ref[i] = (int8_t)(sum / K);
    }

    for (int i = 0; i < M; i++) out[i] = (int8_t)0x5A;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, N, M); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < M; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, M, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
