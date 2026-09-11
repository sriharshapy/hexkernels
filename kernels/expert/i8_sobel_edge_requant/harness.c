#include "harness_common.h"
#include "kernel_api.h"
#define W 66
#define H 34
#define NPIX (W*H)

static uint8_t in[NPIX]  HVX_ALIGN;
static uint8_t out[NPIX] HVX_ALIGN;
static uint8_t ref[NPIX] HVX_ALIGN;

static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
static const int KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};

/* Param sweep: 3 (mult, shift, zp) sets */
static const int MULTS[]  = {  1,  1,  2 };
static const int SHIFTS[] = {  2,  4,  3 };
static const int ZPS[]    = {  0, 10, -5 };
#define NSETS 3

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

static uint8_t ref_pixel(const uint8_t *img, int x, int y,
                         int mult, int shift, int zp){
    int gx=0, gy=0;
    for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
        int p=(int)img[clampi(y+dy,0,H-1)*W + clampi(x+dx,0,W-1)];
        gx += KX[dy+1][dx+1]*p;
        gy += KY[dy+1][dx+1]*p;
    }
    int mag = (gx<0?-gx:gx) + (gy<0?-gy:gy);
    int half = (shift>0) ? (1<<(shift-1)) : 0;
    int v = (mag * mult + half) >> shift;
    v += zp;
    if (v < 0) v=0; if (v>255) v=255;
    return (uint8_t)v;
}

int main(void){
    uint32_t s = 0xFACE0123u;
    for (int i=0; i<NPIX; i++) in[i]=(uint8_t)(hvx_lcg(&s)>>24);
    /* High-contrast horizontal edge */
    for (int x=0; x<W; x++) in[x]=(uint8_t)(x & 0xFF);
    for (int x=0; x<W; x++) in[(H-1)*W+x]=(uint8_t)(255-(x&0xFF));
    /* Vertical edge */
    for (int y=0; y<H; y++) in[y*W+0]=0;
    for (int y=0; y<H; y++) in[y*W+(W-1)]=255;
    /* Checkerboard patch: max both gradients */
    for (int y=20; y<30; y++) for (int x=0; x<W; x++)
        in[y*W+x]=(uint8_t)(((x+y)&1)?255:0);
    /* Flat region: mag=0 */
    for (int y=40; y<50; y++) for (int x=5; x<W-5; x++) in[y*W+x]=128;
    /* w%128 tail: x=64,65 in each row */
    in[5*W+64]=0; in[5*W+65]=255;

    int errors=0, fb=-1; long gotv=0, expv=0;
    for (int k=0; k<NSETS; k++){
        int mult=MULTS[k], shift=SHIFTS[k], zp=ZPS[k];
        for (int i=0; i<NPIX; i++) ref[i]=ref_pixel(in,i%W,i/W,mult,shift,zp);
        for (int i=0; i<NPIX; i++) out[i]=0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i=0; i<NPIX; i++){
            if(out[i]!=ref[i]){ errors++; if(fb<0){fb=k*NPIX+i;gotv=(long)out[i];expv=(long)ref[i];} }
        }
    }
    hvx_report(errors, NPIX*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
