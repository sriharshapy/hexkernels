/* Near-miss: computes 2x2 AVERAGE instead of MAX. Correct shape and tiling, wrong
 * reduction -> differs on every non-constant window. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    int ow = w / 2;
    for (int oy = 0; oy < h/2; oy++)
        for (int ox = 0; ox < ow; ox++) {
            int a=in[(2*oy)*w+2*ox], b=in[(2*oy)*w+2*ox+1];
            int c=in[(2*oy+1)*w+2*ox], d=in[(2*oy+1)*w+2*ox+1];
            out[oy*ow+ox] = (uint8_t)((a+b+c+d)/4);
        }
}
