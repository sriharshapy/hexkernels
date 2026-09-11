/* Near-miss: MAX pool instead of AVERAGE pool (then requant). Correct requant path
 * and runtime-param use, wrong reduction -> differs on most windows. */
#include <stdint.h>
static int8_t ref_element(int pool, int32_t mult, int shift, int8_t zp) {
    int64_t v    = (int64_t)pool * (int64_t)mult;
    int64_t half = shift > 0 ? ((int64_t)1 << (shift - 1)) : 0;
    int64_t r    = (v >= 0) ? ((v + half) >> shift) : -(((-v) + half) >> shift);
    r += zp; if (r > 127) r = 127; if (r < -128) r = -128; return (int8_t)r;
}
void candidate_kernel(const int8_t *in, int8_t *out, int w, int h,
                      int32_t mult, int shift, int8_t zp) {
    int ow = w / 2;
    for (int oy = 0; oy < h/2; oy++)
        for (int ox = 0; ox < ow; ox++) {
            int a=in[(2*oy)*w+2*ox], b=in[(2*oy)*w+2*ox+1];
            int c=in[(2*oy+1)*w+2*ox], d=in[(2*oy+1)*w+2*ox+1];
            int m=a>b?a:b, n=c>d?c:d; int pool=m>n?m:n;   /* MAX instead of avg */
            out[oy*ow+ox] = ref_element(pool, mult, shift, zp);
        }
}
