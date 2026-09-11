#include "harness_common.h"
#include "kernel_api.h"
#include <math.h>

#define NP 100          /* n_pairs -- x has 2*NP=200 elements, not a multiple of 64 */

static hvx_hf x[2*NP]   HVX_ALIGN;
static hvx_hf cosv[NP]  HVX_ALIGN;
static hvx_hf sinv[NP]  HVX_ALIGN;
static hvx_hf out[2*NP] HVX_ALIGN;
static hvx_hf ref[2*NP] HVX_ALIGN;

static float gen_q(uint32_t *s, int range_q, float step) {
    int q = (int)(hvx_lcg(s) % (unsigned)(2*range_q+1)) - range_q;
    return (float)(q * step);
}

int main(void) {
    uint32_t s = 0x900DFEED;

    for (int p = 0; p < NP; p++) {
        x[2*p]   = (hvx_hf)gen_q(&s, 8, 0.5f);   /* -4.0 .. 4.0 */
        x[2*p+1] = (hvx_hf)gen_q(&s, 8, 0.5f);
        /* cos/sin in [-1,1], quantized to eighths (exact-ish in fp16) */
        cosv[p] = (hvx_hf)gen_q(&s, 8, 0.125f);
        sinv[p] = (hvx_hf)gen_q(&s, 8, 0.125f);
    }

    /* Edge pairs (override the random fill):
     *  p=0: identity (cos=1, sin=0) -> out should equal x unchanged.
     *  p=1: pure 90-degree rotate (cos=0, sin=1) -> out[d]=-x[d2], out[d2]=x[d]. */
    cosv[0] = (hvx_hf)1.0f; sinv[0] = (hvx_hf)0.0f;
    cosv[1] = (hvx_hf)0.0f; sinv[1] = (hvx_hf)1.0f;

    /* Independent float32 scalar reference. */
    for (int p = 0; p < NP; p++) {
        int d = 2*p, d2 = 2*p+1;
        float xr = (float)x[d], xi = (float)x[d2];
        float c = (float)cosv[p], sn = (float)sinv[p];
        ref[d]  = (hvx_hf)(xr*c - xi*sn);
        ref[d2] = (hvx_hf)(xr*sn + xi*c);
    }

    for (int i = 0; i < 2*NP; i++) *((volatile unsigned short *)&out[i]) = 0xA5A5; /* poison */

    unsigned long long kc = 0;
    HVX_TIME_KERNEL(kc, { candidate_kernel(x, cosv, sinv, out, NP); });
    printf("HVXENV_KCYCLES kernel=%llu\n", kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < 2*NP; i++) {
        unsigned short g = *(unsigned short *)&out[i], e = *(unsigned short *)&ref[i];
        if (!hvx_close_f16bits(g, e)) {
            errors++; if (fb < 0) { fb = i; gotv = (long)g; expv = (long)e; }
        }
    }
    hvx_report_u16(errors, 2*NP, fb, (unsigned short)gotv, (unsigned short)expv);
    return errors ? 1 : 0;
}
