#include <stdint.h>
static int cl(int v, int lo, int hi){ return v < lo ? lo : (v > hi ? hi : v); }
static int maxi(int a, int b){ return a > b ? a : b; }
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    int ow = (w + 1) / 2;
    int oh = (h + 1) / 2;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++){
            int cx = 2 * ox, cy = 2 * oy;
            int m = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++){
                    int v = in[cl(cy+dy,0,h-1)*w + cl(cx+dx,0,w-1)];
                    m = maxi(m, v);
                }
            out[oy*ow + ox] = (uint8_t)m;
        }
}
