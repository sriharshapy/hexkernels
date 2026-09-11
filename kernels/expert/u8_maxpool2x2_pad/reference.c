#include <stdint.h>
static int maxi(int a, int b){ return a > b ? a : b; }
/* Returns pixel value at (r,c) with zero-padding for out-of-bounds. */
static int px(const uint8_t *in, int w, int h, int r, int c){
    if (r < 0 || r >= h || c < 0 || c >= w) return 0;
    return in[r*w + c];
}
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    int ow = (w + 1) / 2;
    int oh = (h + 1) / 2;
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++){
            int r = 2*oy, c = 2*ox;
            int a = px(in,w,h,r,c),   b = px(in,w,h,r,c+1);
            int d = px(in,w,h,r+1,c), e = px(in,w,h,r+1,c+1);
            out[oy*ow + ox] = (uint8_t)maxi(maxi(a,b), maxi(d,e));
        }
}
