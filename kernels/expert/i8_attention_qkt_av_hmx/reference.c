/* Plain scalar FUSED int8 attention QK^T-then-AV (bit-exact, no softmax) --
 * the DENOMINATOR baseline. Same two-stage requant chain as the reference,
 * computed with plain scalar loops (no HVX/HMX). Transcribed literally from
 * harness.c's reference computation. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int f) { int v = f & 0xFFF; if (v & 0x800) v -= 0x1000; return v; }

void candidate_kernel(const uint8_t *Q, const int8_t *K, const int8_t *V,
                       int32_t *out, int S, int D) {
    static int Sc[32*32];

    /* Stage 1: scores = Q . K^T (K row j IS key vector j -- no transpose needed) */
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            int acc = 0;
            for (int d = 0; d < D; d++) acc += (int)Q[i*D+d] * (int)K[j*D+d];
            Sc[i*S + j] = sx12((acc * 17 + 8) >> 4);
        }

    /* Stage 2: acc = Sc . V */
    for (int i = 0; i < S; i++)
        for (int d = 0; d < D; d++) {
            int acc = 0;
            for (int j = 0; j < S; j++) acc += Sc[i*S + j] * (int)V[j*D + d];
            out[i*D + d] = sx12((acc * 17 + 8) >> 4);
        }
}
