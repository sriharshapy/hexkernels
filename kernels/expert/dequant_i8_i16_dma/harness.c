/* dequant_i8_i16_dma harness (v5, L2 multi-mechanism: hvx + dma + vtcm).
 * Bandwidth-bound int8->int16 dequantize at large N (1B read + 2B write =>
 * 3 bytes/elem of DDR traffic). Correctness contract identical to dequant_i8_i16
 * (subtract zp, multiply by scale, arithmetic right shift toward -inf, saturate
 * to int16). A single runtime (zp,scale,shift) set is passed, chosen so
 * (a-zp)*scale fits int16 (halfword lanes); the kernel must still read the
 * params. The HVX reference is cross-checked against the exact int32 scalar
 * reference on a sample. Harness owns main() and maps an identity VTCM
 * translation before the timed call. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>
#include <string.h>

#ifndef N
#define N 393216
#endif
#define ZP    10
#define SCALE 100
#define SHIFT 5

static int8_t  a[N]   HVX_ALIGN;
static int16_t out[N] HVX_ALIGN;
static int16_t ref[N] HVX_ALIGN;

static int16_t ref_scalar(int8_t x, int8_t zp, int32_t scale, int shift) {
    int32_t v = ((int32_t)x - (int32_t)zp) * scale;
    int32_t r = v >> shift;
    if (r > 32767)  r = 32767;
    if (r < -32768) r = -32768;
    return (int16_t)r;
}

/* halfword-lane dequant: (x - zp)*scale >> shift, valid while it fits int16. */
static inline HVX_Vector dequant_half(HVX_Vector xh, HVX_Vector vzp, HVX_Vector vscale, int shift) {
    HVX_Vector d = Q6_Vh_vsub_VhVh(xh, vzp);
    HVX_Vector p = Q6_Vh_vmpyi_VhVh(d, vscale);
    return Q6_Vh_vasr_VhR(p, shift);
}
/* One input byte-vector (128 int8) -> pair of int16 result vectors in NATURAL
 * order. Q6_Wh_vsxt_Vb DEALs (lo=even elems, hi=odd), so re-interleave the
 * per-lane results with vshuff before storing. */
static inline HVX_VectorPair dequant_bytevec(HVX_Vector xb, HVX_Vector vzp, HVX_Vector vscale, int shift) {
    HVX_VectorPair xh = Q6_Wh_vsxt_Vb(xb);
    HVX_Vector rlo = dequant_half(Q6_V_lo_W(xh), vzp, vscale, shift);
    HVX_Vector rhi = dequant_half(Q6_V_hi_W(xh), vzp, vscale, shift);
    return Q6_W_vshuff_VVR(rhi, rlo, -2);
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    /* HVX fill of a[]. */
    {
        const int vlen = sizeof(HVX_Vector);
        int8_t vi[128], vs[128];
        for (int j = 0; j < 128; j++) { vi[j] = (int8_t)(j*13+1); vs[j] = (int8_t)(128*13); }
        HVX_Vector cur = *(HVX_Vector *)vi, step = *(HVX_Vector *)vs;
        int i = 0;
        for (; i + vlen <= N; i += vlen) { *(HVX_Vector *)(a + i) = cur; cur = Q6_Vb_vadd_VbVb(cur, step); }
        for (; i < N; i++) a[i] = (int8_t)(i*13+1);
    }
    a[0]=-128; a[1]=127; a[2]=0; a[3]=-1; a[4]=10; a[5]=-100;

    uint32_t sw = ((uint32_t)(SCALE & 0xFFFF) << 16) | (SCALE & 0xFFFF);
    uint32_t zw = ((uint32_t)((ZP) & 0xFFFF) << 16) | ((ZP) & 0xFFFF);
    HVX_Vector vscale = Q6_V_vsplat_R(sw);
    HVX_Vector vzp    = Q6_V_vsplat_R(zw);
    {
        const int vlen = sizeof(HVX_Vector);   /* 128 int8 per iter */
        int i = 0;
        for (; i + vlen <= N; i += vlen) {
            HVX_VectorPair r = dequant_bytevec(*(const HVX_Vector *)(a + i), vzp, vscale, SHIFT);
            *(HVX_Vector *)(ref + i)      = Q6_V_lo_W(r);
            *(HVX_Vector *)(ref + i + 64) = Q6_V_hi_W(r);
        }
        for (; i < N; i++) ref[i] = ref_scalar(a[i], ZP, SCALE, SHIFT);
    }
    {
        int bad = 0;
        for (int i = 0; i < N; i += 89) if (ref[i] != ref_scalar(a[i], ZP, SCALE, SHIFT)) { bad = 1; break; }
        for (int i = 0; i < 8; i++) if (ref[i] != ref_scalar(a[i], ZP, SCALE, SHIFT)) bad = 1;
        if (bad) { printf("HVXENV_REF_MISMATCH (harness bug)\n"); return 2; }
    }
    /* Poison out (int16). */
    {
        const int hpv = sizeof(HVX_Vector) / 2;   /* 64 halfwords */
        int16_t pa[64]; for (int j=0;j<64;j++) pa[j] = (int16_t)0xA5A5u;
        HVX_Vector vp = *(HVX_Vector *)pa;
        int i = 0;
        for (; i + hpv <= N; i += hpv) *(HVX_Vector *)(out + i) = vp;
        for (; i < N; i++) out[i] = (int16_t)0xA5A5u;
    }

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(a, out, N, (int8_t)ZP, (int32_t)SCALE, SHIFT); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    /* Byte-wise HVX verify over 2*N bytes; scalar rescan on mismatch. */
    int errors = 0, fb = -1;
    {
        const int vlen = sizeof(HVX_Vector);
        const int8_t *ob = (const int8_t *)out, *rb = (const int8_t *)ref;
        const int nbytes = N * 2;
        int mismatch = 0, i = 0;
        for (; i + vlen <= nbytes && !mismatch; i += vlen) {
            HVX_VectorPred eq = Q6_Q_vcmp_eq_VbVb(*(HVX_Vector *)(ob + i), *(HVX_Vector *)(rb + i));
            int8_t tmp[128] HVX_ALIGN; *(HVX_Vector *)tmp = Q6_V_vand_QR(eq, -1);
            for (int j = 0; j < vlen; j++) if ((uint8_t)tmp[j] != 0xFF) { mismatch = 1; break; }
        }
        if (mismatch) for (int i2 = 0; i2 < N; i2++) if (out[i2] != ref[i2]) { errors++; if (fb < 0) fb = i2; }
    }
    hvx_report(errors, N, fb, fb>=0?(long)out[fb]:0, fb>=0?(long)ref[fb]:0);
    return errors ? 1 : 0;
}
