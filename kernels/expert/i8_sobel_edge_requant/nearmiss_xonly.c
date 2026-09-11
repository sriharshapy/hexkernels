/* Near-miss: computes only the horizontal Sobel gradient (Gx) -- omits Gy.
   mag = |Gx| only. Fails wherever Gy is significant (vertical edges,
   diagonal edges, checkerboard patch). */
#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h,
                      int mult, int shift, int zp){
    for (int y=0; y<h; y++) for (int x=0; x<w; x++){
        int gx=0;
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
            int p=(int)in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
            gx += KX[dy+1][dx+1]*p;
        }
        /* BUG: only |Gx|, Gy omitted */
        int mag = (gx<0?-gx:gx);
        int half = (shift>0) ? (1<<(shift-1)) : 0;
        int v = (mag * mult + half) >> shift;
        v += zp;
        if(v<0) v=0; if(v>255) v=255;
        out[y*w+x]=(uint8_t)v;
    }
}
