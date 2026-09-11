#include "harness_common.h"
#include "kernel_api.h"
#define W 48
#define H 32
#define NPIX (W*H)
static uint8_t in[NPIX] HVX_ALIGN, out[NPIX] HVX_ALIGN, ref[NPIX] HVX_ALIGN;

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
static const int KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

int main(void){
    uint32_t s = 0x5555AAAAu;
    for(int i=0;i<NPIX;i++) in[i]=(uint8_t)(hvx_lcg(&s)>>24);
    /* high-contrast patterns to exercise saturation and both gradient directions */
    for(int x=0;x<W;x++) in[x]=(uint8_t)(x & 0xFF);           /* horizontal ramp -> strong Gx */
    for(int x=0;x<W;x++) in[(H-1)*W+x]=(uint8_t)(255-(x&0xFF));
    for(int y=0;y<H;y++) in[y*W+0]=(uint8_t)0;
    for(int y=0;y<H;y++) in[y*W+(W-1)]=(uint8_t)255;
    /* checkerboard patch at rows 40-50 -> strong both Gx and Gy */
    for(int y=40;y<50;y++) for(int x=0;x<W;x++)
        in[y*W+x]=(uint8_t)(((x+y)&1)?255:0);
    /* w%128 tail */
    in[20*W+128]=0; in[20*W+129]=255;

    /* reference */
    for(int y=0;y<H;y++) for(int x=0;x<W;x++){
        int gx=0, gy=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int p=(int)in[clampi(y+dy,0,H-1)*W + clampi(x+dx,0,W-1)];
            gx += KX[dy+1][dx+1]*p;
            gy += KY[dy+1][dx+1]*p;
        }
        int mag=(gx<0?-gx:gx)+(gy<0?-gy:gy);
        ref[y*W+x]=(uint8_t)(mag>255?255:mag);
    }
    for(int i=0;i<NPIX;i++) out[i]=0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors=0, fb=-1;
    for(int i=0;i<NPIX;i++) if(out[i]!=ref[i]){ errors++; if(fb<0) fb=i; }
    hvx_report(errors, NPIX, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
