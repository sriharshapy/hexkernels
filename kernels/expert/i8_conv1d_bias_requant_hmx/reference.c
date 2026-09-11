/* Plain scalar int8 DENSE 1D conv + bias + int8 requant -- DENOMINATOR baseline.
 * Same fused epilogue (sign-extend 12-bit requant field, add per-channel int32
 * bias, saturate to int8) as the reference, computed with plain scalar loops
 * (no HVX/HMX). Transcribed literally from harness.c's reference computation. */
#include "kernel_api.h"
#include "harness_common.h"
#include <stdint.h>

static inline int sx12(int field) {
    int v = field & 0xFFF;
    if (v & 0x800) v -= 0x1000;
    return v;
}

void candidate_kernel(const uint8_t *in, const int8_t *W, const int32_t *bias,
                      int8_t *out, int n) {
    for (int co = 0; co < C_OUT; co++)
        for (int ol = 0; ol < OL; ol++) {
            int acc = 0;
            for (int ci = 0; ci < C_IN; ci++)
                for (int kw = 0; kw < KW; kw++)
                    acc += (int)in[ci*L_IN + ol + kw] * (int)W[(co*C_IN + ci)*KW + kw];
            int v = sx12(hvx_hmx_requant_0x40(acc)) + bias[co];
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            out[co*OL + ol] = (int8_t)v;
        }
    (void)n;
}
