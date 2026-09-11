/* Near-miss: computes avg pool instead of max pool (wrong reduction). */
#include <stdint.h>
static int cl(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    int ow = (w+1)/2, oh = (h+1)/2;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++){
            int cx = 2*ox, cy = 2*oy;
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += in[cl(cy+dy,0,h-1)*w + cl(cx+dx,0,w-1)];
            out[oy*ow+ox] = (uint8_t)(sum / 9);
        }
}
