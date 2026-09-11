/* Near-miss: uses "valid" output size w/2 x h/2 (truncation, no ceil) instead of
   ceil output size (w+1)/2 x (h+1)/2.  Skips the last output row and column when
   input dims are odd — produces fewer output pixels, causing mismatch at those positions.
   This is also wrong if the caller expects (w+1)/2 x (h+1)/2 output. */
#include <stdint.h>
static int maxi(int a,int b){ return a>b?a:b; }
void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h){
    int ow = w / 2;   /* BUG: truncate instead of (w+1)/2 — skips last col when w odd */
    int oh = h / 2;   /* BUG: truncate instead of (h+1)/2 — skips last row when h odd */
    for (int oy = 0; oy < oh; oy++)
        for (int ox = 0; ox < ow; ox++){
            int a = in[(2*oy)*w   + 2*ox];
            int b = in[(2*oy)*w   + 2*ox+1];
            int c = in[(2*oy+1)*w + 2*ox];
            int d = in[(2*oy+1)*w + 2*ox+1];
            out[oy*ow+ox] = (uint8_t)maxi(maxi(a,b), maxi(c,d));
        }
    /* Last column (ox=ow) and last row (oy=oh) are NOT written — remain poisoned. */
}
