#include "harness_common.h"
#include "kernel_api.h"
/* W=129 odd → ow=65 (tail), H=97 odd → oh=49. */
#define W 129
#define H 97
#define OW ((W+1)/2)
#define OH ((H+1)/2)
static uint8_t in[W*H] HVX_ALIGN, out[OW*OH] HVX_ALIGN, ref[OW*OH] HVX_ALIGN;
static int clamp_i(int v,int lo,int hi){ return v<lo?lo:(v>hi?hi:v); }
int main(void){
    uint32_t s = 0xC0FFEE01u;
    for (int i = 0; i < W*H; i++) in[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* inject: all-255 block at rows 2-4, cols 2-4 → window at ox=1,oy=1 (cx=2,cy=2) = 255/255 */
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            in[(2+dy)*W+(2+dx)] = 255;
    /* inject: all-0 block at last row/col to probe border clamp */
    for (int x = 0; x < W; x++) in[(H-1)*W+x] = 0;
    for (int oy = 0; oy < OH; oy++)
        for (int ox = 0; ox < OW; ox++){
            int cx = 2*ox, cy = 2*oy;
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += in[clamp_i(cy+dy,0,H-1)*W + clamp_i(cx+dx,0,W-1)];
            ref[oy*OW+ox] = (uint8_t)(sum / 9);
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
