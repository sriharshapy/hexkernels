#include "harness_common.h"
#include "kernel_api.h"
#define W 130
#define H 98
#define NPIX (W*H)
static uint8_t  in[NPIX]  HVX_ALIGN;
static int16_t  out[NPIX] HVX_ALIGN;
static int16_t  ref[NPIX] HVX_ALIGN;

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

int main(void){
    uint32_t s = 0xBEEF1234u;
    for(int i=0;i<NPIX;i++) in[i]=(uint8_t)(hvx_lcg(&s)>>24);
    /* border edge cases */
    for(int x=0;x<W;x++) in[x]=(uint8_t)(x & 0xFF);           /* top row gradient */
    for(int x=0;x<W;x++) in[(H-1)*W+x]=(uint8_t)200;
    for(int y=0;y<H;y++) in[y*W+0]=(uint8_t)0;
    for(int y=0;y<H;y++) in[y*W+(W-1)]=(uint8_t)255;
    /* flat region: Laplacian should be 0 */
    for(int y=20;y<30;y++) for(int x=5;x<W-5;x++) in[y*W+x]=(uint8_t)128;
    /* step edge: exercises full range output */
    for(int y=40;y<50;y++){
        for(int x=0;x<W/2;x++) in[y*W+x]=(uint8_t)0;
        for(int x=W/2;x<W;x++) in[y*W+x]=(uint8_t)255;
    }
    /* w%128 tail */
    in[60*W+128]=50; in[60*W+129]=200;

    /* reference */
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int c  = (int)in[y*W+x];
        int up = (int)in[clampi(y-1,0,H-1)*W + x];
        int dn = (int)in[clampi(y+1,0,H-1)*W + x];
        int lt = (int)in[y*W + clampi(x-1,0,W-1)];
        int rt = (int)in[y*W + clampi(x+1,0,W-1)];
        ref[y*W+x]=(int16_t)(up + dn + lt + rt - 4*c);
    }
    for(int i=0;i<NPIX;i++) out[i]=(int16_t)0x5A5A;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for(int i=0;i<NPIX;i++) if(out[i]!=ref[i]){ errors++; if(fb<0) fb=i; }
    hvx_report(errors, NPIX, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
