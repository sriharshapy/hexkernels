/* u8_alpha_blend_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound uint8 alpha blend at large N. Correctness contract identical to
 * u8_alpha_blend: out=(alpha*a+(256-alpha)*b+128)>>8. A single runtime alpha is
 * passed (chosen so alpha*(a-b)+128 fits int16 halfword lanes; the kernel must
 * still read it). The HVX reference uses the identity
 *   out = b + ((alpha*(a-b)+128) >> 8)   (floor shift)
 * and is cross-checked against the exact scalar formula on a sample. Harness owns
 * main() and maps an identity VTCM translation before the timed call. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 393216   /* 3072 vectors; a+b+out = 1.15MB > 1MB L2 -> DDR-bound */
#endif
#define ALPHA 127

static uint8_t a[N]   HVX_ALIGN;
static uint8_t b[N]   HVX_ALIGN;
static uint8_t out[N] HVX_ALIGN;
static uint8_t ref[N] HVX_ALIGN;

static uint8_t ref_scalar(uint8_t ai, uint8_t bi, uint16_t alpha) {
    int v = ((int)alpha*(int)ai + (256-(int)alpha)*(int)bi + 128) >> 8;
    return (uint8_t)v;
}

/* HVX halfword blend (valid while alpha*(a-b)+128 fits int16). */
static inline HVX_Vector blend_vec(HVX_Vector va, HVX_Vector vb, HVX_Vector valpha,
                                   HVX_Vector v128) {
    HVX_VectorPair ah = Q6_Wuh_vzxt_Vub(va);
    HVX_VectorPair bh = Q6_Wuh_vzxt_Vub(vb);
    HVX_Vector al[2] = { Q6_V_lo_W(ah), Q6_V_hi_W(ah) };
    HVX_Vector bl[2] = { Q6_V_lo_W(bh), Q6_V_hi_W(bh) };
    HVX_Vector res[2];
    for (int k = 0; k < 2; k++) {
        HVX_Vector t = Q6_Vh_vsub_VhVh(al[k], bl[k]);            /* a-b in [-255,255] */
        HVX_Vector X = Q6_Vh_vmpyi_VhVh(t, valpha);              /* alpha*(a-b) */
        HVX_Vector y = Q6_Vh_vasr_VhR(Q6_Vh_vadd_VhVh(X, v128), 8);
        res[k] = Q6_Vh_vadd_VhVh(y, bl[k]);                      /* + b, in [0,255] */
    }
    return Q6_Vub_vasr_VhVhR_sat(res[1], res[0], 0);            /* unsigned sat pack */
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vai[128], vbi[128], vas[128], vbs[128];
        for (int j = 0; j < 128; j++) {
            vai[j] = (int8_t)(j*5+1); vbi[j] = (int8_t)(j*9+7);
            vas[j] = (int8_t)(128*5); vbs[j] = (int8_t)(128*9);
        }
        HVX_Vector ca=*(HVX_Vector*)vai, cb=*(HVX_Vector*)vbi, sa=*(HVX_Vector*)vas, sb=*(HVX_Vector*)vbs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            *(HVX_Vector *)(a + i) = ca; *(HVX_Vector *)(b + i) = cb;
            ca = Q6_Vb_vadd_VbVb(ca, sa); cb = Q6_Vb_vadd_VbVb(cb, sb);
        }
        for (; i < N; i++) { a[i]=(uint8_t)(i*5+1); b[i]=(uint8_t)(i*9+7); }
    }
    a[0]=255; b[0]=0;   a[1]=0; b[1]=255;   a[2]=200; b[2]=200;   a[3]=1; b[3]=2;

    uint32_t aw = ((uint32_t)ALPHA << 16) | (uint32_t)ALPHA;
    uint32_t hw = (128u << 16) | 128u;
    HVX_Vector valpha = Q6_V_vsplat_R(aw);
    HVX_Vector v128   = Q6_V_vsplat_R(hw);
    {
        const int vlen = sizeof(HVX_Vector);
        int i = 0;
        for (; i + vlen <= N; i += vlen)
            *(HVX_Vector *)(ref + i) = blend_vec(*(const HVX_Vector *)(a + i),
                                                 *(const HVX_Vector *)(b + i), valpha, v128);
        for (; i < N; i++) ref[i] = ref_scalar(a[i], b[i], ALPHA);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 91) if (ref[i] != ref_scalar(a[i], b[i], ALPHA)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], b[i], ALPHA)) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    {
        const int vlen = sizeof(HVX_Vector);
        uint8_t pa[128]; memset(pa, 0xA5, 128); HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + vlen <= N; i += vlen) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = 0xA5;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, b, out, N, (uint16_t)ALPHA); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        int mismatch = 0, i = 0;
        for (; i + vlen <= N && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(out + i), *(HVX_Vector *)(ref + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        for (; i < N; i++) if (out[i] != ref[i]) { mismatch = 1; break; }
        if (mismatch) for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
