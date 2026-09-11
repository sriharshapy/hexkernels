/* DENOMINATOR baseline -- plain scalar C, no HVX. Straightforward causal
 * self-attention + residual + LUT-FFN + residual, exactly per kernel_api.h. */
#include "kernel_api.h"
#include "harness_common.h"

void candidate_kernel(const uint8_t *X,
                      const int8_t *W1, const int32_t *b1,
                      const int8_t *act_lut,
                      const int8_t *W2, const int32_t *b2,
                      const uint8_t *exp_lut,
                      int8_t *out, int S, int D, int Dff) {
    static int8_t  scaled[32*32] HVX_ALIGN;
    static uint8_t probs[32*32]  HVX_ALIGN;
    static uint8_t h[32*64]      HVX_ALIGN;
    static int8_t  Hact[32*128]  HVX_ALIGN;

    for (int i = 0; i < S; i++) {
        for (int j = 0; j <= i; j++) {
            int raw = 0;
            for (int d = 0; d < D; d++) raw += (int)X[i*D+d] * (int)X[j*D+d];
            scaled[i*S+j] = (int8_t)dec_clamp_i8(raw >> ATTN_SCALE_SHIFT);
        }
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
        for (int d = 0; d < D; d++) {
            int av = 0;
            for (int j = 0; j <= i; j++) av += (int)probs[i*S+j] * (int)X[j*D+d];
            int a = dec_clamp_i8(av >> ATTN_OUT_SHIFT);
            h[i*D+d] = (uint8_t)dec_clamp_u8((int)X[i*D+d] + a);
        }
    }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < Dff; j++) {
            int acc1 = 0;
            for (int k = 0; k < D; k++) acc1 += (int)h[i*D+k] * (int)W1[k*Dff+j];
            int p1 = (acc1 >> FFN_SH1) + b1[j];
            int idx = dec_clamp_i8(p1) + 128;
            Hact[i*Dff+j] = act_lut[idx];
        }
    for (int i = 0; i < S; i++)
        for (int j = 0; j < D; j++) {
            int acc2 = 0;
            for (int k = 0; k < Dff; k++) acc2 += (int)Hact[i*Dff+k] * (int)W2[k*D+j];
            int p2 = (acc2 >> FFN_SH2) + b2[j];
            int fv = dec_sat_i8(p2);
            out[i*D+j] = dec_sat_i8((int)h[i*D+j] + fv);
        }
}
