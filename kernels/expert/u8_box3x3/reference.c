#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    for (int y=0;y<h;y++) for (int x=0;x<w;x++){
        int sum=0;
        for (int dy=-1;dy<=1;dy++) for (int dx=-1;dx<=1;dx++)
            sum += in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
        out[y*w+x]=(uint8_t)(sum/9);
    }
}
