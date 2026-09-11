#include <stdint.h>
static int cl(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
/* Sobel-x kernel: [[-1,0,1],[-2,0,2],[-1,0,1]] with clamp-to-edge. */
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h){
    static const int KX[3][3] = {{-1,0,1},{-2,0,2},{-1,0,1}};
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        int sum=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
            sum += KX[dy+1][dx+1] * (int)in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
        out[y*w+x]=(int16_t)sum;
    }
}
