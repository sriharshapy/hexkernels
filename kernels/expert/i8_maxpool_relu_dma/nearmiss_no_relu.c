/* Near-miss: 2x2 signed max pool WITHOUT the ReLU. Passes on all-positive windows
 * but leaves negative maxima negative where the reference clamps to 0. */
#include <stdint.h>
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h) {
    int ow = w / 2;
    for (int oy = 0; oy < h/2; oy++)
        for (int ox = 0; ox < ow; ox++) {
            int a=in[(2*oy)*w+2*ox], b=in[(2*oy)*w+2*ox+1];
            int c=in[(2*oy+1)*w+2*ox], d=in[(2*oy+1)*w+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; out[oy*ow+ox]=(int8_t)(m>n?m:n);  /* no ReLU */
        }
}
