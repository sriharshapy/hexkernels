/* i8_nchw_to_nhwc harness (v5, L2: hvx + dma + vtcm). C=4 channel planes ->
 * interleaved NHWC. Harness owns main(): HVX-fills the 4 planes, builds the
 * reference with the same 2-level vshuff interleave, cross-checks the first
 * pixel-block against an independent scalar gather (semantic guard), poisons
 * out, times the candidate (kernel-only pcycles), bit-exact compares. Maps VTCM
 * identity so a DMA candidate can stage planes on-chip. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define C 4
#ifndef HW
#define HW 196608            /* per-plane bytes (1536 blocks of 128); in=out=768KB */
#endif
#define NTOT (C*HW)

static int8_t in[NTOT]  HVX_ALIGN;
static int8_t out[NTOT] HVX_ALIGN;
static int8_t ref[NTOT] HVX_ALIGN;

static void interleave4(const int8_t *pin, int8_t *o) {
    const HVX_Vector *p0=(const HVX_Vector*)(pin+0*HW), *p1=(const HVX_Vector*)(pin+1*HW);
    const HVX_Vector *p2=(const HVX_Vector*)(pin+2*HW), *p3=(const HVX_Vector*)(pin+3*HW);
    HVX_Vector *vo=(HVX_Vector*)o;
    int nblk=HW/128, k=0;
    for (; k<nblk; k++) {
        HVX_Vector v0=p0[k],v1=p1[k],v2=p2[k],v3=p3[k];
        HVX_VectorPair a=Q6_W_vshuff_VVR(v1,v0,-1), b=Q6_W_vshuff_VVR(v3,v2,-1);
        HVX_VectorPair c=Q6_W_vshuff_VVR(Q6_V_lo_W(b),Q6_V_lo_W(a),-2);
        HVX_VectorPair d=Q6_W_vshuff_VVR(Q6_V_hi_W(b),Q6_V_hi_W(a),-2);
        vo[4*k+0]=Q6_V_lo_W(c); vo[4*k+1]=Q6_V_hi_W(c);
        vo[4*k+2]=Q6_V_lo_W(d); vo[4*k+3]=Q6_V_hi_W(d);
    }
    for (int p=nblk*128;p<HW;p++) for(int cc=0;cc<4;cc++) o[p*4+cc]=pin[cc*HW+p];
}

int main(void) {
    uint32_t vtcm=__rdcfg(__vtcm_base)<<16;
    add_translation((void*)(uintptr_t)vtcm,(void*)(uintptr_t)vtcm,0);

    /* HVX fill of the 4 planes. */
    {
        const int vlen=128;
        for (int c=0;c<C;c++) {
            int8_t iv[128], sv[128];
            for (int j=0;j<128;j++){ iv[j]=(int8_t)(j*3 + c*31 + 1); sv[j]=(int8_t)(128*3); }
            HVX_Vector cur=*(HVX_Vector*)iv, st=*(HVX_Vector*)sv;
            int8_t *pl=in+c*HW; int i=0;
            for (; i+vlen<=HW; i+=vlen){ *(HVX_Vector*)(pl+i)=cur; cur=Q6_Vb_vadd_VbVb(cur,st); }
            for (; i<HW; i++) pl[i]=(int8_t)(i*3 + c*31 + 1);
        }
    }
    in[0]=127; in[HW]=-128; in[2*HW]=1; in[3*HW]=-1;   /* discriminators in each plane */

    interleave4(in, ref);
    /* Independent scalar cross-check on the first 128 pixels. */
    for (int p=0;p<128;p++) for(int c=0;c<4;c++)
        if (ref[p*4+c] != in[c*HW+p]) { printf("HVXENV_REFCHECK_FAIL p=%d c=%d\n",p,c); return 2; }

    { uint8_t pa[128]; for(int j=0;j<128;j++)pa[j]=0xA5; HVX_Vector vp=*(HVX_Vector*)pa;
      int i=0; for(;i+128<=NTOT;i+=128)*(HVX_Vector*)(out+i)=vp; for(;i<NTOT;i++)out[i]=(int8_t)0xA5; }

    unsigned long long _hvx_kc=0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, C, HW); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors=0, fb=-1;
    for (int i=0;i<NTOT;i++) if(out[i]!=ref[i]){ errors++; if(fb<0)fb=i; }
    hvx_report(errors, NTOT, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors?1:0;
}
