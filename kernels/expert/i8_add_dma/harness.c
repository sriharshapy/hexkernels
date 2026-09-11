/* i8_add_dma harness (L1, single-mechanism ladder rung: hvx + dma + vtcm).
 * Bandwidth-bound int8 saturating add at large N (2 bytes read + 1 byte write
 * per element). Correctness contract: out[i] = sat_i8(a[i] + b[i]). Harness
 * owns main() and maps an identity VTCM translation before the timed call
 * (the single-buffer expert stages tiles through VTCM via uDMA; the scalar
 * baseline and near-miss ignore it). */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 262144
#endif

static int8_t a[N]   HVX_ALIGN;
static int8_t b[N]   HVX_ALIGN;
static int8_t out[N] HVX_ALIGN;
static int8_t ref[N] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        uint32_t seed = 0x0ADDAu;
        for (int i = 0; i < N; i++) a[i] = (int8_t)(hvx_lcg(&seed) & 0xFFu);
    }
    {
        uint32_t seed = 0x0ADDBu;
        for (int i = 0; i < N; i++) b[i] = (int8_t)(hvx_lcg(&seed) & 0xFFu);
    }
    a[0] = 127;  b[0] = 127;   /* -> saturate to 127 */
    a[1] = -128; b[1] = -128;  /* -> saturate to -128 */
    a[2] = 0;    b[2] = 0;     /* -> 0 */

    for (int i = 0; i < N; i++) {
        int s = (int)a[i] + (int)b[i];
        if (s > 127) s = 127;
        if (s < -128) s = -128;
        ref[i] = (int8_t)s;
    }

    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
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
