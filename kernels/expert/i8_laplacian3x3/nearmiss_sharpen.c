/* Near-miss: implements the sharpening kernel [[0,-1,0],[-1,5,-1],[0,-1,0]]
   clamped to uint8 — same 4-connected stencil but wrong coefficients (+5 vs -4
   center, negative neighbours vs positive) and wrong output type/saturation.
   Will fail on virtually every pixel that is not uniform. */
#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h){
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        int c  = (int)in[y*w+x];
        int up = (int)in[cl(y-1,0,h-1)*w + x];
        int dn = (int)in[cl(y+1,0,h-1)*w + x];
        int lt = (int)in[y*w + cl(x-1,0,w-1)];
        int rt = (int)in[y*w + cl(x+1,0,w-1)];
        /* BUG: sharpening formula instead of Laplacian */
        int v  = 5*c - up - dn - lt - rt;
        out[y*w+x]=(int16_t)(v<0?0:(v>255?255:v));
    }
}
