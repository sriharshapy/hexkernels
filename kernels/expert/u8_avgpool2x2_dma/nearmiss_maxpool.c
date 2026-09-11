/* Near-miss: computes 2x2 MAX instead of truncated AVERAGE. Correct shape, wrong
 * reduction -> differs on every non-constant window. */
#include <stdint.h>
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h) {
    int ow = w / 2;
    for (int oy = 0; oy < h/2; oy++)
        for (int ox = 0; ox < ow; ox++) {
            int a=in[(2*oy)*w+2*ox], b=in[(2*oy)*w+2*ox+1];
            int c=in[(2*oy+1)*w+2*ox], d=in[(2*oy+1)*w+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; out[oy*ow+ox]=(uint8_t)(m>n?m:n);
        }
}
