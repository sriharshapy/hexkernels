#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      const int8_t *weights, int32_t bias,
                      int32_t mult, int shift, int8_t zp){
    for (int y=0; y<h; y++) for (int x=0; x<w; x++){
        int32_t acc=0;
        for (int dy=-1; dy<=1; dy++) for (int dx=-1; dx<=1; dx++){
            int ny=cl(y+dy,0,h-1), nx=cl(x+dx,0,w-1);
            acc += (int32_t)weights[(dy+1)*3+(dx+1)] * (int32_t)in[ny*w+nx];
        }
        int64_t v = (int64_t)acc + (int64_t)bias;
        int64_t half = (shift>0) ? ((int64_t)1<<(shift-1)) : 0;
        int64_t r;
        if (v >= 0) r = (v*(int64_t)mult + half) >> shift;
        else        r = -(((-v)*(int64_t)mult + half) >> shift);
        r += (int64_t)zp;
        if (r > 127) r=127; if(r<-128) r=-128;
        out[y*w+x]=(int8_t)r;
    }
}
