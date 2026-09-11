/* EXPERT (achievability bar) -- combines s1 (vectorized d=x-mu subtract)
 * and s2 (vectorized SUM reduction via sign-extend widen) into one
 * pipeline: HVX for the row SUM (mu) and for the elementwise d=x-mu
 * subtract; var reduction and the gamma/inv/beta epilogue math stay
 * scalar (this ISA has no direct elementwise 32x16->32 vector*vector
 * multiply without a widen-multiply-combine sequence, so that step is
 * left scalar -- pure integer, bit-exact regardless). */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

#define ROWCAP 128

void candidate_kernel(const int16_t *x, int16_t *out, int R, int C,
                      const int16_t *gamma, const int16_t *beta,
                      const uint16_t *inv_lut) {
    for (int r = 0; r < R; r++) {
        const int16_t *xr = x + (long)r * C;
        int16_t *outr = out + (long)r * C;

        int16_t xrow[ROWCAP] __attribute__((aligned(128)));
        int16_t drow[ROWCAP] __attribute__((aligned(128)));
        for (int c = 0; c < C; c++) xrow[c] = xr[c];

        /* Vectorized SUM (mu). */
        HVX_Vector accLo = Q6_V_vzero(), accHi = Q6_V_vzero();
        int c = 0;
        for (; c + 64 <= C; c += 64) {
            HVX_Vector xv = *(const HVX_Vector *)(xrow + c);
            HVX_VectorPair wp = Q6_Ww_vsxt_Vh(xv);
            accLo = Q6_Vw_vadd_VwVw(accLo, Q6_V_lo_W(wp));
            accHi = Q6_Vw_vadd_VwVw(accHi, Q6_V_hi_W(wp));
        }
        int32_t lo[32] __attribute__((aligned(128)));
        int32_t hi[32] __attribute__((aligned(128)));
        *(HVX_Vector *)lo = accLo;
        *(HVX_Vector *)hi = accHi;
        int64_t sum = 0;
        for (int j = 0; j < 32; j++) sum += (int64_t)lo[j] + (int64_t)hi[j];
        for (; c < C; c++) sum += (int64_t)xrow[c];

        int32_t mu = (int32_t)(sum / C);

        /* Vectorized elementwise d = x - mu. */
        HVX_Vector muVec = Q6_Vh_vsplat_R(mu);
        c = 0;
        for (; c + 64 <= C; c += 64) {
            HVX_Vector xv = *(const HVX_Vector *)(xrow + c);
            *(HVX_Vector *)(drow + c) = Q6_Vh_vsub_VhVh(xv, muVec);
        }
        for (; c < C; c++) drow[c] = (int16_t)(xrow[c] - mu);

        /* var (scalar, from the already-vectorized d). */
        int64_t var_sum = 0;
        for (int cc = 0; cc < C; cc++) {
            int64_t d = (int64_t)drow[cc];
            var_sum += d * d;
        }
        int64_t var = var_sum / C;
        int32_t vidx = (int32_t)(var >> 5);
        if (vidx < 0) vidx = 0;
        if (vidx > 255) vidx = 255;
        int32_t inv = (int32_t)inv_lut[vidx];

        for (int cc = 0; cc < C; cc++) {
            int32_t d      = (int32_t)drow[cc];
            int32_t scaled = (d * (int32_t)gamma[cc] + 32) >> 6;
            int32_t normed = (scaled * inv + 512) >> 10;
            int32_t res    = normed + (int32_t)beta[cc];
            if (res >  32767) res =  32767;
            if (res < -32768) res = -32768;
            outr[cc] = (int16_t)res;
        }
    }
}
