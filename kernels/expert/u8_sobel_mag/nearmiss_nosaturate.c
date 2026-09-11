/* Near-miss: forgets to clamp the magnitude to 255 (uint8 wraps instead).
   When |Gx|+|Gy| > 255 (common on high-contrast edges / checkerboard),
   the truncated cast will produce a wrong value instead of 255. */
#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
static const int KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        int gx=0, gy=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++){
            int p=(int)in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
            gx += KX[dy+1][dx+1]*p;
            gy += KY[dy+1][dx+1]*p;
        }
        /* BUG: no clamp — wraps on high-contrast edges */
        int mag=(gx<0?-gx:gx)+(gy<0?-gy:gy);
        out[y*w+x]=(uint8_t)mag;
    }
}
