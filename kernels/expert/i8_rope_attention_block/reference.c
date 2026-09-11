/* DENOMINATOR baseline -- the WHOLE RoPE attention block in plain scalar C,
 * no HVX, no HMX matrix engine. Transcribed directly from the harness's
 * independent scalar golden (rope_attn_ref in harness.c):
 *   RoPE  : per-position rotation of Q,K feature pairs
 *   QK^T  : scalar row-column dot products
 *   scale+softmax : fixed-point LUT, row-wise over the key axis
 *   A.V   : scalar row-column dot products
 * The same 0x40-config requant as the reference is applied after each matmul. */
#include <stdint.h>
#include "kernel_api.h"
#include "harness_common.h"

static inline int sx12(int field) {
    int v = field & 0xFFF; if (v & 0x800) v -= 0x1000; return v;
}
static inline int clamp_i8(int v) {
    if (v >  127) return  127; if (v < -128) return -128; return v;
}
static inline int clamp_u8(int v) {
    if (v > 255) return 255; if (v < 0) return 0; return v;
}

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                      const int8_t *cos_lut, const int8_t *sin_lut,
                      uint16_t *out, int S, int D, int rot_shift,
                      const uint8_t *exp_lut) {
    static uint8_t Qr[64*64];
    static int8_t  Kr[64*64];
    static int8_t  scaled[64*64];
    static uint8_t probs[64*64];

    const int HALF = D / 2;

    /* Stage 0: RoPE rotation of Q,K over feature pairs (d, d+HALF). */
    for (int i = 0; i < S; i++) {
        for (int d = 0; d < HALF; d++) {
            int c = cos_lut[i*HALF+d], s = sin_lut[i*HALF+d];
            int q0 = Q[i*D+d], q1 = Q[i*D+d+HALF];
            Qr[i*D+d]      = (uint8_t)clamp_u8((q0*c - q1*s) >> rot_shift);
            Qr[i*D+d+HALF] = (uint8_t)clamp_u8((q0*s + q1*c) >> rot_shift);
            int k0 = K[i*D+d], k1 = K[i*D+d+HALF];
            Kr[i*D+d]      = (int8_t)clamp_i8((k0*c - k1*s) >> rot_shift);
            Kr[i*D+d+HALF] = (int8_t)clamp_i8((k0*s + k1*c) >> rot_shift);
        }
    }

    /* Stage 1: scores = Qr.Kr^T, HMX 0x40 requant, then scale + clamp to int8 */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Qr[i*D+d] * (int)Kr[j*D+d];
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
        for (int j = 0; j < S; j++)
            probs[i*S+j] = (uint8_t)(((int)probs[i*S+j] * 255 + half) / Sr);
    }

    /* Stage 3: out = probs . V, HMX 0x40 requant */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += (int)probs[i*S+j] * (int)V[j*D+d];
            out[i*D + d] = (uint16_t)hvx_hmx_requant_0x40(acc);
        }
}
