#include "harness_common.h"
#include "kernel_api.h"
#define W 34
#define H 34
#define NPIX (W*H)

static int8_t  in[NPIX]  HVX_ALIGN;
static int8_t  out[NPIX] HVX_ALIGN;
static int8_t  ref[NPIX] HVX_ALIGN;

static int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

/* Param sweep: 3 sets of (weights, bias, mult, shift, zp) */
/* Kernel A: Laplacian-style [-1,-1,-1,-1,8,-1,-1,-1,-1] */
static const int8_t WEIGHTS_A[9] = {-1,-1,-1,-1,8,-1,-1,-1,-1};
/* Kernel B: horizontal edge [-1,-2,-1,0,0,0,1,2,1] */
static const int8_t WEIGHTS_B[9] = {-1,-2,-1,0,0,0,1,2,1};
/* Kernel C: sharpening [0,-1,0,-1,5,-1,0,-1,0] */
static const int8_t WEIGHTS_C[9] = {0,-1,0,-1,5,-1,0,-1,0};

static const int32_t BIASES[]  = {   0,  16,  -8 };
static const int32_t MULTS[]   = {   1,   2,   1 };
static const int     SHIFTS[]  = {   3,   4,   2 };
static const int8_t  ZPS[]     = {   0,  -5,   3 };

#define NSETS 3

static int8_t ref_pixel(const int8_t *img, int x, int y,
                        const int8_t *wts, int32_t bias,
                        int32_t mult, int shift, int8_t zp){
    int32_t acc=0;
    for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
        int ny=clampi(y+dy,0,H-1), nx=clampi(x+dx,0,W-1);
        acc += (int32_t)wts[(dy+1)*3+(dx+1)] * (int32_t)img[ny*W+nx];
    }
    int64_t v = (int64_t)acc + (int64_t)bias;
    int64_t half = (shift>0) ? ((int64_t)1<<(shift-1)) : 0;
    int64_t r;
    if (v >= 0) r = (v*(int64_t)mult + half) >> shift;
    else        r = -(((-v)*(int64_t)mult + half) >> shift);
    r += (int64_t)zp;
    if (r >  127) r= 127;
    if (r < -128) r=-128;
    return (int8_t)r;
}

int main(void){
    uint32_t s = 0xC0FFEE42u;
    for (int i=0; i<NPIX; i++) in[i]=(int8_t)(hvx_lcg(&s)>>24);
    /* Edge cases */
    for (int x=0; x<W; x++) in[x]=(int8_t)(x-W/2);       /* top row ramp */
    for (int x=0; x<W; x++) in[(H-1)*W+x]=(int8_t)64;    /* bottom row constant */
    /* Flat patch: stencil output depends only on bias+requant */
    for (int y=10; y<15; y++) for (int x=2; x<W-2; x++) in[y*W+x]=(int8_t)16;
    /* High-contrast step: exercises saturation */
    for (int y=20; y<25; y++){
        for (int x=0; x<W/2; x++) in[y*W+x]=(int8_t)-128;
        for (int x=W/2; x<W; x++) in[y*W+x]=(int8_t)127;
    }
    /* w%128 tail: x=32,33 */
    in[5*W+32]=(int8_t)100; in[5*W+33]=(int8_t)-100;

    static const int8_t *WEIGHTS_SET[3] = {WEIGHTS_A, WEIGHTS_B, WEIGHTS_C};

    int errors=0, fb=-1; long gotv=0, expv=0;
    for (int k=0; k<NSETS; k++){
        const int8_t *wts=WEIGHTS_SET[k];
        int32_t bias=BIASES[k], mult=MULTS[k];
        int shift=SHIFTS[k]; int8_t zp=ZPS[k];
        for (int i=0; i<NPIX; i++) ref[i]=ref_pixel(in,i%W,i/W,wts,bias,mult,shift,zp);
        for (int i=0; i<NPIX; i++) out[i]=(int8_t)0xA5;
        unsigned long long _hvx_kc = 0;
        HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H, wts, bias, mult, shift, zp); });
        printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
        for (int i=0; i<NPIX; i++){
            if(out[i]!=ref[i]){ errors++; if(fb<0){fb=k*NPIX+i;gotv=(long)out[i];expv=(long)ref[i];} }
        }
    }
    hvx_report(errors, NPIX*NSETS, fb, gotv, expv);
    return errors ? 1 : 0;
}
