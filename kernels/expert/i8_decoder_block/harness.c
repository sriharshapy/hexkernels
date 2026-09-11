/* i8_decoder_block harness (L3 composite, holdout group H).
 * Causal self-attention + residual1 + LUT-activation FFN + residual2, all int8
 * fixed-point. The harness owns main(): seeds deterministic inputs, computes the
 * FULL composed scalar reference (plain integer loops -- no HVX needed for a
 * reference; bit-exactness of a vectorized candidate follows from integer
 * add/mul associativity as long as it sums exactly the causal window and no more),
 * poisons the output, times the candidate (kernel-only pcycles), and does a
 * bit-exact compare. */
#include "harness_common.h"
#include "kernel_api.h"

#define S   32
#define D   64
#define DFF 128

static uint8_t X [S*D]     HVX_ALIGN;   /* input activations, uint8 0..3 */
static int8_t  W1[D*DFF]   HVX_ALIGN;   /* up-proj weights               */
static int32_t b1[DFF]     HVX_ALIGN;
static int8_t  act_lut[256] HVX_ALIGN;  /* runtime FFN activation LUT    */
static int8_t  W2[DFF*D]   HVX_ALIGN;   /* down-proj weights             */
static int32_t b2[D]       HVX_ALIGN;
static uint8_t exp_lut[256] HVX_ALIGN;  /* runtime fixed-point exp LUT   */

static int8_t  scaled[S*S] HVX_ALIGN;   /* causal scores (upper tri unused) */
static uint8_t probs[S*S]  HVX_ALIGN;
static uint8_t h [S*D]     HVX_ALIGN;   /* residual-stream FFN activation */
static int8_t  Hact[S*DFF] HVX_ALIGN;
static int8_t  out[S*D]    HVX_ALIGN;
static int8_t  ref[S*D]    HVX_ALIGN;

static void decoder_ref(void) {
    for (int i = 0; i < S; i++) {
        /* causal scores j in [0,i] */
        for (int j = 0; j <= i; j++) {
            int raw = 0;
            for (int d = 0; d < D; d++) raw += (int)X[i*D+d] * (int)X[j*D+d];
            scaled[i*S+j] = (int8_t)dec_clamp_i8(raw >> ATTN_SCALE_SHIFT);
        }
        /* causal softmax over j in [0,i] using exp_lut */
        int m = scaled[i*S+0];
        for (int j = 1; j <= i; j++) if (scaled[i*S+j] > m) m = scaled[i*S+j];
        int Sr = 0;
        for (int j = 0; j <= i; j++) {
            int diff = (int)scaled[i*S+j] - m;
            if (diff < -255) diff = -255;
            uint8_t e = exp_lut[diff + 255];
            probs[i*S+j] = e;
            Sr += (int)e;
        }
        int half = Sr / 2;
        for (int j = 0; j <= i; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
        /* attention output: weighted V over the causal window only */
        for (int d = 0; d < D; d++) {
            int av = 0;
            for (int j = 0; j <= i; j++) av += (int)probs[i*S+j] * (int)X[j*D+d];
            int a = dec_clamp_i8(av >> ATTN_OUT_SHIFT);
            h[i*D+d] = (uint8_t)dec_clamp_u8((int)X[i*D+d] + a);   /* residual 1 */
        }
    }
    /* FFN: up-proj -> LUT activation -> down-proj -> residual 2 */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < DFF; j++) {
            int acc1 = 0;
            for (int k = 0; k < D; k++) acc1 += (int)h[i*D+k] * (int)W1[k*DFF+j];
            int p1 = (acc1 >> FFN_SH1) + b1[j];
            int idx = dec_clamp_i8(p1) + 128;
            Hact[i*DFF+j] = act_lut[idx];
        }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = 0;
            for (int k = 0; k < DFF; k++) acc2 += (int)Hact[i*DFF+k] * (int)W2[k*D+j];
            int p2 = (acc2 >> FFN_SH2) + b2[j];
            int fv = dec_sat_i8(p2);
            ref[i*D+j] = dec_sat_i8((int)h[i*D+j] + fv);
        }
}

int main(void) {
    uint32_t s = 0xD3C0DE11u;

    /* X uint8 in 0..3; W1 in -3..3; W2 in -2..2. */
    for (int i = 0; i < S*D;   i++) X[i]  = (uint8_t)(hvx_lcg(&s) % 4);
    for (int i = 0; i < D*DFF; i++) W1[i] = (int8_t)((int)(hvx_lcg(&s) % 7) - 3);
    for (int i = 0; i < DFF*D; i++) W2[i] = (int8_t)((int)(hvx_lcg(&s) % 5) - 2);
    for (int j = 0; j < DFF; j++) b1[j] = (int32_t)((int)(hvx_lcg(&s) % 129) - 64);
    for (int j = 0; j < D;   j++) b2[j] = (int32_t)((int)(hvx_lcg(&s) % 33) - 16);
    b1[0] = -4000; b1[1] = 4000;    /* LUT-index saturate-low / saturate-high columns */
    b2[0] = -300;  b2[1] = 300;     /* FFN output saturate-low / saturate-high columns */

    /* Row 0: fully causal-degenerate (only attends to itself) -- edge case. */
    X[0*D+0] = 3;

    /* Runtime fixed-point exp LUT (Q16), no libm -- matches i8_transformer_encoder_block. */
    {
        const uint32_t DECAY_Q16 = 62865u;
        uint64_t vq = (uint64_t)255u << 16;
        for (int i = 255; i >= 0; i--) {
            exp_lut[i] = (uint8_t)((vq + 32768u) >> 16);
            vq = (vq * DECAY_Q16) >> 16;
        }
        exp_lut[255] = 255;
    }
    /* Runtime GELU-shaped activation LUT: negative side suppressed, positive
     * side ~identity (saturating), NOT a closed-form ReLU the candidate could
     * reconstruct from a formula -- it must actually index act_lut. */
    for (int k = 0; k < 256; k++) {
        int x = k - 128;                       /* -128..127 */
        int v;
        if (x <= -8)      v = -4;               /* deep-negative: small residual leak */
        else if (x < 0)   v = (x * 3) / 8;      /* soft negative slope                */
        else              v = x - (x*x)/256;    /* soft saturating positive slope      */
        act_lut[k] = dec_sat_i8(v);
    }

    decoder_ref();

    for (int i = 0; i < S*D; i++) *((volatile signed char *)&out[i]) = (signed char)0xA5; /* poison */

    unsigned long long _hvx_kc = 0;
    HVX_TIME_KERNEL(_hvx_kc, { candidate_kernel(X, W1, b1, act_lut, W2, b2, exp_lut, out, S, D, DFF); });
    printf("HVXENV_KCYCLES kernel=%llu\n", _hvx_kc);

    int errors = 0, fb = -1; long gotv = 0, expv = 0;
    for (int i = 0; i < S*D; i++) {
        int g = (int)out[i], e = (int)ref[i];
        if (g != e) { errors++; if (fb < 0) { fb = i; gotv = g; expv = e; } }
    }
    hvx_report(errors, S*D, fb, gotv, expv);
    return errors ? 1 : 0;
}
