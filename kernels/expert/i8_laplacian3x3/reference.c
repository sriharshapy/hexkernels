#include <stdint.h>
static int cl(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }
/* Laplacian kernel [[0,1,0],[1,-4,1],[0,1,0]] with clamp-to-edge. */
void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h){
    for(int y=0;y<h;y++) for(int x=0;x<w;x++){
        int c  = (int)in[y*w+x];
        int up = (int)in[cl(y-1,0,h-1)*w + x];
        int dn = (int)in[cl(y+1,0,h-1)*w + x];
        int lt = (int)in[y*w + cl(x-1,0,w-1)];
        int rt = (int)in[y*w + cl(x+1,0,w-1)];
        out[y*w+x]=(int16_t)(up + dn + lt + rt - 4*c);
    }
}
