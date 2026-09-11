/* i8_interleave harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * 2-way interleave (zip): out[2i]=a[i], out[2i+1]=b[i]. Two concurrent input
 * streams make the direct-DDR path latency-bound; a double-buffered DMA+VTCM
 * expert recovers the gap (>=1.2x). Harness owns main(): HVX-fills a[],b[],
 * builds the reference with HVX vshuff, cross-checks the first vector against an
 * independent scalar zip (semantic guard), poisons out, times the candidate
 * (kernel-only pcycles), bit-exact compares. Maps VTCM identity so a DMA
 * candidate can stage tiles on-chip. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef N
#define N 393216            /* per-stream elems; out=2N=768KB, working set ~1.5MB > L2 */
#endif
#define NOUT (2*N)

static int8_t a[N]      HVX_ALIGN;
static int8_t b[N]      HVX_ALIGN;
static int8_t out[NOUT] HVX_ALIGN;
static int8_t ref[NOUT] HVX_ALIGN;

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* HVX fill of a[] and b[] (deterministic, cheap). */
    {
        const int vlen = 128;
        int8_t ia[128], ib[128], sa[128], sb[128];
        for (int j = 0; j < 128; j++) {
            ia[j] = (int8_t)(j * 5 + 1);  ib[j] = (int8_t)(j * 9 + 2);
            sa[j] = (int8_t)(128 * 5);    sb[j] = (int8_t)(128 * 9);
        }
        HVX_Vector ca = *(HVX_Vector*)ia, cb = *(HVX_Vector*)ib;
        HVX_Vector ta = *(HVX_Vector*)sa, tb = *(HVX_Vector*)sb;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector*)(a+i) = ca; *(HVX_Vector*)(b+i) = cb;
            ca = Q6_Vb_vadd_VbVb(ca, ta); cb = Q6_Vb_vadd_VbVb(cb, tb);
        }
        for (; i < N; i++) { a[i]=(int8_t)(i*5+1); b[i]=(int8_t)(i*9+2); }
    }
    /* Boundary/discriminating values. */
    a[0]=127; b[0]=-128; a[1]=-1; b[1]=1; a[N-1]=42; b[N-1]=-42;

    /* Reference: HVX vshuff interleave (fast). */
    {
        const HVX_Vector *va=(const HVX_Vector*)a, *vb=(const HVX_Vector*)b;
        HVX_Vector *vo=(HVX_Vector*)ref;
        int nv = N/128, k=0;
        for (; k < nv; k++) {
            HVX_VectorPair p = Q6_W_vshuff_VVR(vb[k], va[k], -1);
            vo[2*k]=Q6_V_lo_W(p); vo[2*k+1]=Q6_V_hi_W(p);
        }
        for (int i = nv*128; i < N; i++) { ref[2*i]=a[i]; ref[2*i+1]=b[i]; }
    }
    /* Independent scalar cross-check on the first 128 pairs (semantic guard). */
    for (int i = 0; i < 128 && i < N; i++) {
        if (ref[2*i] != a[i] || ref[2*i+1] != b[i]) {
            printf("HVXENV_REFCHECK_FAIL i=%d\n", i);
            return 2;
        }
    }
    /* Poison. */
    {
        uint8_t pa[128]; for (int j=0;j<128;j++) pa[j]=0xA5;
        HVX_Vector vp=*(HVX_Vector*)pa;
        int i=0; for (; i+128<=NOUT; i+=128) *(HVX_Vector*)(out+i)=vp;
        for (; i<NOUT; i++) out[i]=(int8_t)0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1;
    for (int i=0;i<NOUT;i++) if (out[i]!=ref[i]) { errors++; if(fb<0)fb=i; }
    hvx_report(errors, NOUT, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors?1:0;
}
