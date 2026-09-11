/* Near-miss: computes Sobel-y instead of Sobel-x.
   Uses kernel [[-1,-2,-1],[0,0,0],[1,2,1]] — detects horizontal edges (y-gradient)
   instead of the required vertical-edge x-gradient. Will fail on gradient input. */
#include <stdint.h>
static int cl(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
static const int KY[3][3] = {{-1,-2,-1},{0,0,0},{1,2,1}};
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h){
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        int sum=0;
        for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
            sum += KY[dy+1][dx+1]*(int)in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
        out[y*w+x]=(int16_t)sum;
    }
}
