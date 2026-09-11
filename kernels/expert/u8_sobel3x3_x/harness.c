#include "harness_common.h"
#include "kernel_api.h"
#define W 48
#define H 32
#define NPIX (W*H)
static uint8_t  in[NPIX]  HVX_ALIGN;
static int16_t  out[NPIX] HVX_ALIGN;
static int16_t  ref[NPIX] HVX_ALIGN;

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};

int main(void){
    uint32_t s = 0xABCD1234u;
    for(int i=0;i<NPIX;i++) in[i]=(uint8_t)(hvx_lcg(&s)>>24);
    /* injected border cases: high-contrast edges exercise Sobel heavily */
    for(int x=0;x<W;x++) in[x]=(uint8_t)(x & 0xFF);         /* top row gradient -> strong Gx */
    for(int x=0;x<W;x++) in[(H-1)*W+x]=(uint8_t)200;
    for(int y=0;y<H;y++) in[y*W+0]=(uint8_t)0;
    for(int y=0;y<H;y++) in[y*W+(W-1)]=(uint8_t)255;
    /* w%128 tail */
    in[7*W+128]=10; in[7*W+129]=245;

    /* reference */
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int sum=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
            sum += KX[dy+1][dx+1]*(int)in[clampi(y+dy,0,H-1)*W + clampi(x+dx,0,W-1)];
        ref[y*W+x]=(int16_t)sum;
    }
    for(int i=0;i<NPIX;i++) out[i]=(int16_t)0x5A5A;  /* poison */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for(int i=0;i<NPIX;i++) if(out[i]!=ref[i]){ errors++; if(fb<0) fb=i; }
    hvx_report(errors, NPIX, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
