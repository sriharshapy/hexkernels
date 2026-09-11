/* DENOMINATOR baseline -- the WHOLE attention block, plain scalar C, no HVX/HMX.
 *   QK^T : scalar row-row dot products
 *   scale+softmax : fixed-point LUT, row-wise over the key axis
 *   A.V  : scalar row-row dots (probs uint8 . V int8)
 * The same 0x40-config requant as the reference is applied after each matmul.
 * Transcribed literally from harness.c's attention_ref (the oracle). */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v;
}
static inline int clamp_i8(int v) {
    if (v > 127) return 127; if (v < -128) return -128; return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      uint16_t *out, int S, int D, const uint8_t *exp_lut) {
    static int8_t  scaled[64*64];
    static uint8_t probs[64*64];

    /* Stage 1: scores = Q.K^T, HMX 0x40 requant, then scale + clamp to int8 */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Q[i*D+d] * (int)K[j*D+d];
            int s12 = hvx_hmx_requant_0x40(acc);
            scaled[i*S+j] = (int8_t)clamp_i8(sx12(s12) >> 4);
        }

    /* Stage 2: row-wise softmax over the KEY axis j (fixed-point LUT) */
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
        for (int j = 0; j < S; j++) {
            int ej = (int)probs[i*S+j];
            probs[i*S+j] = (uint8_t)((ej * 255 + half) / Sr);
        }
    }

    /* Stage 3: out = probs . V, HMX 0x40 requant */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += (int)probs[i*S+j] * (int)V[j*D+d];
            out[i*D+d] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
}
