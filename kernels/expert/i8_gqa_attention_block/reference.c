/* DENOMINATOR baseline -- the WHOLE grouped-query attention block, pure scalar
 * (no HVX/HMX matrix engine). Each query head is processed independently
 * against its shared KV head via plain nested loops:
 *   QK^T          : scalar row-row dot products
 *   scale+softmax : fixed-point LUT, row-wise over the key axis (scalar reduce)
 *   A.V           : scalar probs(uint8).V row-row dots
 * Same 0x40-config requant as the reference after each matmul. */
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
                      uint16_t *out, const uint8_t *exp_lut) {
    static int8_t  scaled[GQA_S*GQA_S];
    static uint8_t probs[GQA_S*GQA_S];

    for (int hq = 0; hq < GQA_H_Q; hq++) {
        int hkv = hq / GQA_GROUP_SIZE;
        const uint8_t *Qh = Q + (size_t)hq  * GQA_S * GQA_D;
        const int8_t  *Kh = K + (size_t)hkv * GQA_S * GQA_D;
        const int8_t  *Vh = V + (size_t)hkv * GQA_S * GQA_D;
        uint16_t      *Oh = out + (size_t)hq * GQA_S * GQA_D;

        /* Stage 1: scores = Q.K^T, requant, scale + clamp -> int8 scaled */
        for (int i = 0; i < GQA_S; i++)
            for (int j = 0; j < GQA_S; j++) {
                int acc = 0;
                for (int d = 0; d < GQA_D; d++) acc += (int)Qh[i*GQA_D+d] * (int)Kh[j*GQA_D+d];
                int s12 = hvx_hmx_requant_0x40(acc);
                scaled[i*GQA_S+j] = (int8_t)clamp_i8(sx12(s12) >> 4);
            }
        /* Stage 2: row-wise softmax over key axis (fixed-point LUT) */
        for (int i = 0; i < GQA_S; i++) {
            int m = scaled[i*GQA_S+0];
            for (int j = 1; j < GQA_S; j++) if (scaled[i*GQA_S+j] > m) m = scaled[i*GQA_S+j];
            int Sr = 0;
            for (int j = 0; j < GQA_S; j++) {
                int diff = (int)scaled[i*GQA_S+j] - m;
                if (diff < -255) diff = -255;
                uint8_t e = exp_lut[diff + 255];
                probs[i*GQA_S+j] = e;
                Sr += (int)e;
            }
            int half = Sr / 2;
            for (int j = 0; j < GQA_S; j++)
                probs[i*GQA_S+j] = (uint8_t)(((int)probs[i*GQA_S+j] * 255 + half) / Sr);
        }
        /* Stage 3: out = probs . V */
        for (int i = 0; i < GQA_S; i++)
            for (int d = 0; d < GQA_D; d++) {
                int acc = 0;
                for (int j = 0; j < GQA_S; j++) acc += (int)probs[i*GQA_S+j] * (int)Vh[j*GQA_D+d];
                Oh[i*GQA_D + d] = (uint16_t)hvx_hmx_requant_0x40(acc);
            }
    }
}
