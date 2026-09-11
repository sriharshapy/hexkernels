#include "harness_common.h"
#include "kernel_api.h"
#define W 48
#define H 32
#define NPIX (W*H)
static uint8_t in[NPIX] HVX_ALIGN, out[NPIX] HVX_ALIGN, ref[NPIX] HVX_ALIGN;

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
int main(void){
    uint32_t s = 0x1A2B3Cu;
    for (int i=0;i<NPIX;i++) in[i]=(uint8_t)(hvx_lcg(&s)>>24);
    for (int x=0;x<W;x++) in[x]=(uint8_t)x;         /* top row gradient (border) */
    for (int i=0;i<W;i++) in[(H-1)*W+i]=200;        /* bottom row constant */
    for (int y=0;y<H;y++) for (int x=0;x<W;x++){
        int sum=0;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
            sum += in[clampi(y+dy,0,H-1)*W + clampi(x+dx,0,W-1)];
        ref[y*W+x]=(uint8_t)(sum/9);
    }
    for (int i=0;i<NPIX;i++) out[i]=0xA5;            /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for (int i=0;i<NPIX;i++) if (out[i]!=ref[i]) { errors++; if(fb<0) fb=i; }
    hvx_report(errors, NPIX, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
