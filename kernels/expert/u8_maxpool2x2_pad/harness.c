#include "harness_common.h"
#include "kernel_api.h"
/* W=129, H=97 (both odd) → ow=(129+1)/2=65, oh=(97+1)/2=49.
   ow=65 not a multiple of 128 (tail).
   Last column of each row and last row are boundary windows touching zero-pad. */
#define W 129
#define H 97
#define OW ((W+1)/2)
#define OH ((H+1)/2)
static uint8_t in[W*H] HVX_ALIGN, out[OW*OH] HVX_ALIGN, ref[OW*OH] HVX_ALIGN;
static int px_ref(const uint8_t *img, int w, int h, int r, int c){
    if (r < 0 || r >= h || c < 0 || c >= w) return 0;
    return img[r*w+c];
}
static int maxi(int a,int b){ return a>b?a:b; }
int main(void){
    uint32_t s = 0xBEEF1234u;
    for (int i = 0; i < W*H; i++) in[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* inject: last column pixel (border window) is 200 — pad is 0 so max stays 200 */
    in[W-1] = 200;
    /* inject: top-left window max = 255 */
    in[0] = 255;
    for (int oy = 0; oy < OH; oy++)
        for (int ox = 0; ox < OW; ox++){
            int r = 2*oy, c = 2*ox;
            int a = px_ref(in,W,H,r,c),   b = px_ref(in,W,H,r,c+1);
            int d = px_ref(in,W,H,r+1,c), e = px_ref(in,W,H,r+1,c+1);
            ref[oy*OW+ox] = (uint8_t)maxi(maxi(a,b), maxi(d,e));
        }
    for (int i = 0; i < OW*OH; i++) out[i] = 0xA5;
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(in, out, W, H); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);
    int errors = 0, fb = -1;
    for (int i = 0; i < OW*OH; i++)
        if (out[i] != ref[i]) { errors++; if (fb < 0) fb = i; }
    hvx_report(errors, OW*OH, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
