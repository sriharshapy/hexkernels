/* dma_nchw_nhwc_i8 harness (v6, Group C: banded gather-DMA-chain NCHW->NHWC
 * layout conversion + vtcm). Harness owns main(): seeds deterministic input,
 * poisons out, maps VTCM identity, times the candidate (kernel-only
 * pcycles), bit-exact compares against a scalar reference.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

/* NOTE: these harness size macros are named IMG_N/IMG_C/IMG_H/IMG_W (NOT
 * N/C/H/W) because N/C/H/W are the candidate_kernel PARAMETER NAMES used
 * inside baseline.c / expert.c / solutions / nearmiss -- a command-line
 * -DN=.. would otherwise clash with those local parameter identifiers when
 * compiled together in one invocation. */
#ifndef IMG_N
#define IMG_N 1
#endif
#ifndef IMG_C
#define IMG_C 64
#endif
#ifndef IMG_H
#define IMG_H 100
#endif
#ifndef IMG_W
#define IMG_W 200
#endif

#define ISZ ((long)IMG_N * IMG_C * IMG_H * IMG_W)

static int8_t in[ISZ]  HVX_ALIGN;
static int8_t out[ISZ] HVX_ALIGN;
static int8_t ref[ISZ] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        uint32_t s = 0x1A2B3C4Du;
        for (long i = 0; i < ISZ; i++) in[i] = (int8_t)hvx_lcg(&s);
    }
    /* Boundary values in the first/last image, channel, and spatial cell. */
    in[0] = 127;
    in[ISZ - 1] = -128;
    in[(long)(IMG_C / 2) * IMG_H * IMG_W + (IMG_H / 2) * IMG_W + (IMG_W / 2)] = 55;

    for (long n = 0; n < IMG_N; n++)
        for (long c = 0; c < IMG_C; c++)
            for (long h = 0; h < IMG_H; h++)
                for (long w = 0; w < IMG_W; w++)
                    ref[((n * IMG_H + h) * IMG_W + w) * IMG_C + c] =
                        in[((n * IMG_C + c) * IMG_H + h) * IMG_W + w];

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128] HVX_ALIGN; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        long i = 0;
        for (; i + vlen <= ISZ; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < ISZ; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, IMG_N, IMG_C, IMG_H, IMG_W); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0; long fb = -1;
    for (long i = 0; i < ISZ; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, (int)ISZ, (int)fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
