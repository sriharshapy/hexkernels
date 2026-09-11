/* i8_transformer_encoder_block harness (v5, L3 composite).
 * Full encoder block: self-attention (X.X^T -> softmax -> A.V) + residual, then FFN
 * (up -> ReLU -> down) + residual, all int8. The harness owns main(): fills seeded
 * inputs, computes the reference via HVX vrmpy for the four matmuls (a fully scalar
 * reference over ~1.6M MACs would blow the sim budget), cross-checks that reference
 * against an INDEPENDENT full scalar golden on row 0 (semantic guard: never trust one
 * implementation), poisons the output, times the candidate (kernel-only pcycles), and
 * does a bit-exact compare. Enables the HMX context AND an identity VTCM translation so
 * a candidate can compose HMX + VTCM + HVX. */
#include "harness_common.h"
#include "kernel_api.h"
#include "hexagon_standalone.h"
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define S   64
#define D   64
#define DFF 128

static uint8_t X [S*D]     HVX_ALIGN;   /* input activations, uint8 0..3 */
static int8_t  Xw[S*D]     HVX_ALIGN;   /* X as int8 weight (positive)   */
static int8_t  Xt[D*S]     HVX_ALIGN;   /* X transposed for A.V vrmpy    */
static int8_t  W1[D*DFF]   HVX_ALIGN;   /* up-proj weights               */
static int8_t  W1t[DFF*D]  HVX_ALIGN;
static int32_t b1[DFF]     HVX_ALIGN;
static int8_t  W2[DFF*D]   HVX_ALIGN;   /* down-proj weights             */
static int8_t  W2t[D*DFF]  HVX_ALIGN;
static int32_t b2[D]       HVX_ALIGN;
static uint8_t exp_lut[256] HVX_ALIGN;

static int8_t  scaled[S*S] HVX_ALIGN;
static uint8_t probs[S*S]  HVX_ALIGN;
static uint8_t h [S*D]     HVX_ALIGN;   /* residual-stream FFN activation */
static uint8_t Href[S*DFF] HVX_ALIGN;
static int8_t  out[S*D]    HVX_ALIGN;
static int8_t  ref[S*D]    HVX_ALIGN;

static inline int32_t hreduce32(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

/* HVX vrmpy reference: uint8 activation row . int8 weight row over `nb` bytes (<=128).
 * Mask only for a partial vector; a full 128-byte reduction uses the whole vector
 * (Q6_Q_vsetq_R only encodes 0..127, so 128 must NOT be masked). */
static inline int32_t vdot(const uint8_t *a, const int8_t *w, int nb) {
    HVX_Vector zero = Q6_V_vzero();
    HVX_Vector va = *(const HVX_UVector *)a;
    HVX_Vector vw = *(const HVX_UVector *)w;
    if (nb < 128) {
        HVX_VectorPred p = Q6_Q_vsetq_R(nb);
        va = Q6_V_vmux_QVV(p, va, zero);
        vw = Q6_V_vmux_QVV(p, vw, zero);
    }
    return hreduce32(Q6_Vw_vrmpyacc_VwVubVb(zero, va, vw));
}

int main(void) {
    uint32_t vtcm = __rdcfg(__vtcm_base) << 16;
    add_translation((void*)(uintptr_t)vtcm, (void*)(uintptr_t)vtcm, 0);

    uint32_t s = 0x71C0FFEEu;
    /* X in 0..3. W1 in -3..3, W2 in -2..2. See kernel_api.h for the field bounds:
     * X.X^T <=576, A.V <=765, up-proj <=1152, down-proj <=1792 -> all *17/16 < 2048,
     * so every HMX 12-bit requant field is exact and the block is bit-exact int8. */
    for (int i = 0; i < S*D;   i++) X[i]  = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < D*DFF; i++) W1[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int i = 0; i < DFF*D; i++) W2[i] = (int8_t)((int)(hvx_lcg(&s) % 5) - 2);
    for (int j = 0; j < DFF; j++) b1[j] = (int32_t)((int)(hvx_lcg(&s) % 129) - 64);
    for (int j = 0; j < D;   j++) b2[j] = (int32_t)((int)(hvx_lcg(&s) % 513) - 256);
    b1[0] = -4000; b1[1] = 4000;    /* FFN ReLU-kill and always-fire columns   */
    b2[0] = -4000; b2[1] = 4000;    /* int8-saturate-low and -high output cols  */

    for (int i = 0; i < S*D; i++) Xw[i] = (int8_t)X[i];
    for (int i = 0; i < S; i++) for (int d = 0; d < D; d++) Xt[d*S + i] = (int8_t)X[i*D + d];
    for (int k = 0; k < D;   k++) for (int j = 0; j < DFF; j++) W1t[j*D + k]   = W1[k*DFF + j];
    for (int k = 0; k < DFF; k++) for (int j = 0; j < D;   j++) W2t[j*DFF + k] = W2[k*D + j];

    /* Fixed-point exp LUT (Q16), no libm. */
    {
        const uint32_t DECAY_Q16 = 62865u;
        uint64_t vq = (uint64_t)255u << 16;
        for (int i = 255; i >= 0; i--) {
            exp_lut[i] = (uint8_t)((vq + 32768u) >> 16);
            vq = (vq * DECAY_Q16) >> 16;
        }
        exp_lut[255] = 255;
    }

    /* ---- Self-attention (HVX vrmpy) ---- */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = (int)vdot(X + i*D, Xw + j*D, D);
            scaled[i*S+j] = (int8_t)tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_SCALE_SHIFT);
        }
    for (int i = 0; i < S; i++) {
        int m = scaled[i*S+0];
        for (int j = 1; j < S; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j < S; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j < S; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
    }
    /* A.V + residual1 -> h */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = (int)vdot(probs + i*S, Xt + d*S, S);
            int a = tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_OUT_SHIFT);
            h[i*D+d] = (uint8_t)tf_clamp_u8((int)X[i*D+d] + a);
        }

    /* ---- FFN (HVX vrmpy) ---- */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < DFF; j++) {
            int acc1 = (int)vdot(h + i*D, W1t + j*D, D);
            int r1 = tf_sx12(hvx_hmx_requant_0x40(acc1));
            Href[i*DFF+j] = (uint8_t)tf_relu_requant(r1 + b1[j]);
        }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = (int)vdot(Href + i*DFF, W2t + j*DFF, DFF);
            int r2 = tf_sx12(hvx_hmx_requant_0x40(acc2));
            int fv = (r2 + b2[j]) >> FFN_SH2;
            ref[i*D+j] = tf_sat_i8((int)h[i*D+j] + fv);
        }

    /* ---- Independent full scalar golden on row 0 (guards the vrmpy semantics) ---- */
    {
        int sc[S], pr[S], hs[D], Hs[DFF];
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)X[0*D+d] * (int)X[j*D+d];
            sc[j] = tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_SCALE_SHIFT);
        }
        int m = sc[0]; for (int j = 1; j < S; j++) if (sc[j] > m) m = sc[j];
        int Sr = 0;
        for (int j = 0; j < S; j++) {
            int diff = sc[j] - m; if (diff < -255) diff = -255;
            pr[j] = exp_lut[diff + 255]; Sr += pr[j];
        }
        int half = Sr / 2;
        for (int j = 0; j < S; j++) pr[j] = (pr[j] * 255 + half) / Sr;
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += pr[j] * (int)X[j*D+d];
            int a = tf_clamp_i8(tf_sx12(hvx_hmx_requant_0x40(acc)) >> ATTN_OUT_SHIFT);
            hs[d] = tf_clamp_u8((int)X[0*D+d] + a);
        }
        for (int j = 0; j < DFF; j++) {
            int acc1 = 0;
            for (int k = 0; k < D; k++) acc1 += hs[k] * (int)W1[k*DFF+j];
            int r1 = tf_sx12(hvx_hmx_requant_0x40(acc1));
            Hs[j] = tf_relu_requant(r1 + b1[j]);
        }
        for (int j = 0; j < D; j++) {
            int acc2 = 0;
            for (int k = 0; k < DFF; k++) acc2 += Hs[k] * (int)W2[k*D+j];
            int r2 = tf_sx12(hvx_hmx_requant_0x40(acc2));
            int fv = (r2 + b2[j]) >> FFN_SH2;
            int g = (int)tf_sat_i8(hs[j] + fv);
            if (g != (int)ref[j]) {
                printf("HVXENV_REFCHECK_FAIL j=%d scalar=%d vrmpy=%d\n", j, g, (int)ref[j]);
                return 2;
            }
        }
    }

    for (int i = 0; i < S*D; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    hvx_hmx_enable();
    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, W1, b1, W2, b2, exp_lut, out, S, D, DFF); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
