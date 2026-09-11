/* i8_ffn_block harness (v5, L3 composition — transformer FFN / MLP block).
 * FFN = matmul1 -> ReLU+requant -> matmul2, two int8 matmuls with a gated
 * activation between (H = ReLU(X*W1+b1), out = H*W2+b2), the feed-forward layer
 * of a transformer. The harness owns main(): fills seeded inputs, computes the
 * reference via HVX vrmpy (a naive scalar S*Dff*D + S*D*Dff reference would blow
 * the sim budget), cross-checks that HVX reference against an INDEPENDENT scalar
 * golden on row 0 (semantic guard: never trust one implementation), poisons the
 * output, times the candidate (kernel-only pcycles), and does a bit-exact compare.
 * Enables the HMX context AND an identity VTCM translation so a candidate can
 * compose HMX + VTCM + HVX. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#ifndef S
#define S 64
#endif
#ifndef D
#define D 128
#endif
#ifndef DFF
#define DFF 128
#endif

static uint8_t X [S*D]     HVX_ALIGN;   /* activations, uint8 0..3        */
static int8_t  W1[D*DFF]   HVX_ALIGN;   /* up-projection weights, int8    */
static int8_t  W1t[DFF*D]  HVX_ALIGN;   /* W1 transposed for vrmpy        */
static int32_t b1[DFF]     HVX_ALIGN;
static int8_t  W2[DFF*D]   HVX_ALIGN;   /* down-projection weights, int8  */
static int8_t  W2t[D*DFF]  HVX_ALIGN;   /* W2 transposed for vrmpy        */
static int32_t b2[D]       HVX_ALIGN;
static int8_t  H [S*DFF]   HVX_ALIGN;   /* reference intermediate, int8   */
static int8_t  out[S*D]    HVX_ALIGN;
static int8_t  ref[S*D]    HVX_ALIGN;

/* Horizontal sum of the 32 word lanes of a vrmpy accumulator. */
static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

int main(void) {
    /* Map VTCM identity so a candidate can stage the intermediate H on-chip. */
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x6E33u;
    /* X in 0..3 (positive: int8==uint8). W1 in -3..3: |acc1| <= D*3*3 = 1152 ->
     * |r1| <= 1224 < 2048 (12-bit field exact). H = ReLU((r1+b1)>>8) in 0..~6.
     * W2 in -2..2: |acc2| <= Dff*7*2 = 1792 -> |r2| <= 1904 < 2048 (exact). */
    for (int i = 0; i < S*D;   i++) X[i]  = (uint8_t)(hvx_lcg(&s) % 4);            /* 0..3 */
    for (int i = 0; i < D*DFF; i++) W1[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3); /* -3..3 */
    for (int i = 0; i < DFF*D; i++) W2[i] = (int8_t)((int)(hvx_lcg(&s) % 5) - 2); /* -2..2 */
    for (int j = 0; j < DFF; j++) b1[j] = (int32_t)((int)(hvx_lcg(&s) % 129) - 64);
    for (int j = 0; j < D;   j++) b2[j] = (int32_t)((int)(hvx_lcg(&s) % 513) - 256);
    b1[0] = -4000; b1[1] = 4000;    /* ReLU-kill and always-fire columns          */
    b2[0] = -4000; b2[1] = 4000;    /* int8-saturate-low and -high output columns  */

    /* transpose: W1t[j*D+k] = W1[k*DFF+j];  W2t[j*DFF+k] = W2[k*D+j] */
    for (int k = 0; k < D;   k++) for (int j = 0; j < DFF; j++) W1t[j*D + k]   = W1[k*DFF + j];
    for (int k = 0; k < DFF; k++) for (int j = 0; j < D;   j++) W2t[j*DFF + k] = W2[k*D + j];

    const HVX_Vector zero = Q6_V_vzero();

    /* --- matmul1 (HVX vrmpy) -> ReLU+requant -> H.  D==128 => one whole vrmpy. --- */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < DFF; j++) {
            HVX_Vector vp = zero;
            for (int k = 0; k < D; k += 128) {
                HVX_Vector va = *(const HVX_UVector *)(X   + i*D + k);
                HVX_Vector vb = *(const HVX_UVector *)(W1t + j*D + k);
                vp = Q6_Vw_vrmpyacc_VwVbVb(vp, va, vb);
            }
            int acc1 = (int)hreduce32(vp);
            int r1   = ffn_sx12((acc1 * 17 + 8) >> 4);
            H[i*DFF + j] = (int8_t)ffn_relu_requant(r1 + b1[j]);
        }

    /* --- matmul2 (HVX vrmpy) -> requant+bias -> saturate.  Dff==128 => one vrmpy. --- */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            HVX_Vector vp = zero;
            for (int k = 0; k < DFF; k += 128) {
                HVX_Vector va = *(const HVX_UVector *)(H   + i*DFF + k);
                HVX_Vector vb = *(const HVX_UVector *)(W2t + j*DFF + k);
                vp = Q6_Vw_vrmpyacc_VwVbVb(vp, va, vb);
            }
            int acc2 = (int)hreduce32(vp);
            int r2   = ffn_sx12((acc2 * 17 + 8) >> 4);
            ref[i*D + j] = ffn_sat_i8((r2 + b2[j]) >> FFN_SH2);
        }

    /* Independent scalar golden on row 0 (full FFN) — guards the vrmpy semantics. */
    {
        int Hs[DFF];
        for (int j = 0; j < DFF; j++) {
            int acc1 = 0;
            for (int k = 0; k < D; k++) acc1 += (int)X[0*D + k] * (int)W1[k*DFF + j];
            int r1 = ffn_sx12((acc1 * 17 + 8) >> 4);
            Hs[j] = ffn_relu_requant(r1 + b1[j]);
        }
        for (int j = 0; j < D; j++) {
            int acc2 = 0;
            for (int k = 0; k < DFF; k++) acc2 += Hs[k] * (int)W2[k*D + j];
            int r2 = ffn_sx12((acc2 * 17 + 8) >> 4);
            int g  = (int)ffn_sat_i8((r2 + b2[j]) >> FFN_SH2);
            if (g != (int)ref[0*D + j]) {
                printf("HVXENV_REFCHECK_FAIL j=%d scalar=%d vrmpy=%d\n", j, g, (int)ref[j]);
                return 2;
            }
        }
    }

    for (int i = 0; i < S*D; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    hvx_hmx_enable();                 /* harness enables HMX; candidate writes compute only */
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, W1, b1, W2, b2, out, S, D, DFF); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
