/* mac_stream_toy harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Toy operand-streaming block-MAC. a=uint8 activations, b=int8 weights; per
 * block dot-product accumulated in int32 (exact). Harness owns main(); maps VTCM
 * identity before the timed call. Two input streams + large N -> bandwidth-bound.
 */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 540672    /* 33 blocks; working set 2*528KB=1.03MB > 1MB L2, sim<=40s */
#endif
#define BLK 16384    /* one block = one tile */
#define NB (N / BLK)

static uint8_t a[N] HVX_ALIGN;
static int8_t  b[N] HVX_ALIGN;
static int32_t out[NB] HVX_ALIGN;
static int32_t ref[NB] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* HVX fill of a[] (uint8) and b[] (int8). */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t ai[128], bi[128], as[128], bs[128];
        for (int j = 0; j < 128; j++) { ai[j]=(int8_t)(j*5+1); bi[j]=(int8_t)(j*3+2); as[j]=(int8_t)(128*5); bs[j]=(int8_t)(128*3); }
        HVX_Vector ca=*(HVX_Vector*)ai, cb=*(HVX_Vector*)bi, sa=*(HVX_Vector*)as, sb=*(HVX_Vector*)bs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector*)(a+i)=ca; *(HVX_Vector*)(b+i)=cb;
            ca=Q6_Vb_vadd_VbVb(ca,sa); cb=Q6_Vb_vadd_VbVb(cb,sb);
        }
        for (; i < N; i++) { a[i]=(uint8_t)(i*5+1); b[i]=(int8_t)(i*3+2); }
    }
    /* Edge values: max magnitudes. */
    a[0]=255; b[0]=127; a[1]=255; b[1]=-128; a[BLK]=0; b[BLK]=-128;

    /* Exact int32 reference: per-block dot product. */
    for (int m = 0; m < NB; m++) {
        int32_t s = 0;
        for (int k = 0; k < BLK; k++) s += (int32_t)a[m*BLK+k] * (int32_t)b[m*BLK+k];
        ref[m] = s;
    }
    for (int m = 0; m < NB; m++) out[m] = (int32_t)0xA5A5A5A5;   /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N, BLK); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    for (int m = 0; m < NB; m++) if (out[m] != ref[m]) { errors++; if (fb < 0) fb = m; }
    hvx_report(errors, NB, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
