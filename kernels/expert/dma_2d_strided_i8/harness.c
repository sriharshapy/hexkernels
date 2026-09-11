/* dma_2d_strided_i8 harness (v6, Group C: 2D/row-strided dma + vtcm).
 * Depads an H x rowstride buffer into a compact H x W buffer. Harness owns
 * main(): seeds deterministic input including the row-padding bytes (which
 * must NOT leak into out[]), poisons out, maps VTCM identity, times the
 * candidate (kernel-only pcycles), bit-exact compares.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef HROWS
#define HROWS 136
#endif
#ifndef WCOLS
#define WCOLS 8192
#endif
#ifndef ROWSTRIDE
#define ROWSTRIDE 8320   /* WCOLS + 128 bytes of padding */
#endif

#define ASZ ((long)HROWS * ROWSTRIDE)
#define OSZ ((long)HROWS * WCOLS)

static int8_t a[ASZ]   HVX_ALIGN;
static int8_t out[OSZ] HVX_ALIGN;
static int8_t ref[OSZ] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* Fill a[] including the padding bytes with a deterministic pattern;
     * the padding bytes are poison-like sentinels (0x5A) that must never
     * appear in out[] -- a candidate that accidentally copies rowstride
     * bytes instead of W bytes per row will leak them and fail. */
    {
        uint32_t s = 0x13572468u;
        for (long r = 0; r < HROWS; r++) {
            int8_t *row = a + r * ROWSTRIDE;
            for (int c = 0; c < WCOLS; c++) row[c] = (int8_t)hvx_lcg(&s);
            for (int c = WCOLS; c < ROWSTRIDE; c++) row[c] = (int8_t)0x5A;
        }
    }
    /* Boundary rows/cols. */
    a[0] = 127; a[1] = -128;
    a[(HROWS-1)*ROWSTRIDE + WCOLS - 1] = 100;
    a[(HROWS-1)*ROWSTRIDE + WCOLS - 2] = -100;

    for (long r = 0; r < HROWS; r++)
        for (int c = 0; c < WCOLS; c++) ref[r * WCOLS + c] = a[r * ROWSTRIDE + c];

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t poison_arr[128]; memset(poison_arr, 0xA5, 128);
        HVX_Vector vpois = *(HVX_Vector *)poison_arr;
        long i = 0;
        for (; i + vlen <= OSZ; i += vlen) *(HVX_Vector *)(out + i) = vpois;
        for (; i < OSZ; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, HROWS, WCOLS, ROWSTRIDE); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0; long fb = -1;
    for (long i = 0; i < OSZ; i++) if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, (int)OSZ, (int)fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
