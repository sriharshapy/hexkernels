/* Near-miss: computes average pool instead of L2 pool (omits the sqrt and squares). */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    int ow = w / 2;
    for (int oy = 0; oy < h/2; oy++)
        for (int ox = 0; ox < ow; ox++){
            int a = in[(2*oy)*w   + 2*ox];
            int b = in[(2*oy)*w   + 2*ox+1];
            int c = in[(2*oy+1)*w + 2*ox];
            int d = in[(2*oy+1)*w + 2*ox+1];
            out[oy*ow+ox] = (uint8_t)((a + b + c + d) / 4);
        }
}
