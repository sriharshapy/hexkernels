/* dma_double_buffer_mul_i16 harness (v6, Group C: dma + vtcm double-buffer).
 * int16 low-truncating multiply at large N. Harness owns main(): seeds
 * deterministic inputs with overflow-boundary values, poisons out, maps
 * VTCM identity, times the candidate (kernel-only pcycles), bit-exact
 * compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 270056   /* int16 elems; 540112B/array; a+b+out ~1.6MB > L2 */
#endif

static int16_t a[N]   HVX_ALIGN;
static int16_t b[N]   HVX_ALIGN;
static int16_t out[N] HVX_ALIGN;
static int16_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x9E3779B9u;
    for (int i = 0; i < N; i++) { a[i] = (int16_t)hvx_lcg(&s); b[i] = (int16_t)hvx_lcg(&s); }
    /* Overflow-boundary values. */
    a[0] = 32767;  b[0] = 32767;   /* overflow */
    a[1] = -32768; b[1] = -32768;  /* overflow */
    a[2] = -32768; b[2] = 1;       /* no overflow, negative */
    a[N-2] = 100;  b[N-2] = 400;   /* small positive */
    a[N-1] = -1;   b[N-1] = -1;

    for (int i = 0; i < N; i++) ref[i] = (int16_t)((int32_t)a[i] * (int32_t)b[i]);

    {
        const int vlen = sizeof(HVX_Vector) / sizeof(int16_t);
        int16_t poison_arr[64] HVX_ALIGN;
        for (int j = 0; j < 64; j++) poison_arr[j] = (int16_t)0xA5A5;
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < N; i++) out[i] = (int16_t)0xA5A5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int i = 0; i < N; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
