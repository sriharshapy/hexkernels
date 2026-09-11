#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>
/* W=130, H=98 → OW=65 (not multiple of 128, exercises tail). */
#define W 130
#define H 98
#define OW (W/2)
#define OH (H/2)
static uint8_t in[W*H] HVX_ALIGN, out[OW*OH] HVX_ALIGN, ref[OW*OH] HVX_ALIGN;
int main(void){
    uint32_t s = 0x1357ACEFu;
    for (int i = 0; i < W*H; i++) in[i] = (uint8_t)(hvx_lcg(&s) >> 24);
    /* inject: first window a=b=c=d=127 → L2=127*2=254 (below saturation) */
    in[0]=127; in[1]=127; in[W]=127; in[W+1]=127;
    /* inject: a window with all-255 values → sqrt(4*255^2)=510 → saturate to 255 */
    in[2]=255; in[3]=255; in[W+2]=255; in[W+3]=255;
    for (int oy = 0; oy < OH; oy++)
        for (int ox = 0; ox < OW; ox++){
            int a = in[(2*oy)*W   + 2*ox];
            int b = in[(2*oy)*W   + 2*ox+1];
            int c = in[(2*oy+1)*W + 2*ox];
            int d = in[(2*oy+1)*W + 2*ox+1];
            uint32_t sv = (uint32_t)sqrtf((float)(a*a + b*b + c*c + d*d));
            ref[oy*OW+ox] = (uint8_t)(sv > 255 ? 255 : sv);
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
