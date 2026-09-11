/* EXPERT (achievability bar) -- same as solutions/s1.c: HVX max-reduce +
 * vlut32 gather + HVX vrmpyacc sum; scalar epilogue for recip lookup /
 * final multiply-shift-clamp (8-100 scalar values per row). C=100 fits in
 * a single 128-lane vector (padded), so there is no full-vector/tail split
 * within a row -- the tail-path property of this task instead comes from
 * C not being a multiple of 128 (28 padding lanes per row must be excluded
 * from the max and the sum). */
#include "kernel_api.h"
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

static inline int8_t hvec_hmax_b(HVX_Vector v) {
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 4));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 2));
    v = Q6_Vb_vmax_VbVb(v, Q6_V_vror_VR(v, 1));
    return (int8_t)Q6_R_vextract_VR(v, 0);
}

static inline int32_t hvec_hsum_w(HVX_Vector v) {
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 64));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 32));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 16));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 8));
    v = Q6_Vw_vadd_VwVw(v, Q6_V_vror_VR(v, 4));
    return (int32_t)Q6_R_vextract_VR(v, 0);
}

void candidate_kernel(const int8_t *x, uint8_t *out, int R, int C,
                      const uint8_t *exp_lut, const uint16_t *recip_lut)
{
    /* Preprocess 256-entry exp_lut into two shuffled half-tables for vlut32. */
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(exp_lut));
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(exp_lut + 128));

    uint8_t xbuf[128]   __attribute__((aligned(128)));
    uint8_t idx_buf[128] __attribute__((aligned(128)));

    for (int r = 0; r < R; r++) {
        const int8_t *row  = x   + r * C;
        uint8_t      *orow = out + r * C;

        /* ---- row max via HVX byte-max reduction (pad with -128, neutral) ---- */
        for (int j = 0; j < C; j++) xbuf[j] = (uint8_t)row[j];
        for (int j = C; j < 128; j++) xbuf[j] = (uint8_t)(-128);
        HVX_Vector vx = *(const HVX_Vector *)xbuf;
        int8_t m = hvec_hmax_b(vx);

        /* ---- idx derivation + HVX vlut32 8-pass gather for e_j ---- */
        for (int j = 0; j < C; j++) {
            int diff = (int)row[j] - (int)m;
            if (diff < -255) diff = -255;
            if (diff > 0) diff = 0;
            idx_buf[j] = (uint8_t)(diff + 255);
        }
        for (int j = C; j < 128; j++) idx_buf[j] = 255; /* placeholder; zeroed below */
        HVX_Vector vidx = *(const HVX_Vector *)idx_buf;

        HVX_Vector vexp;
        vexp = Q6_Vb_vlut32_VbVbR(vidx, sTab0, 0);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab0, 1);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab0, 2);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab0, 3);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab1, 4);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab1, 5);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab1, 6);
        vexp = Q6_Vb_vlut32or_VbVbVbR(vexp, vidx, sTab1, 7);

        union { HVX_Vector v; uint8_t b[128]; } eu;
        eu.v = vexp;
        for (int j = C; j < 128; j++) eu.b[j] = 0;  /* exclude padding lanes from sum */

        /* ---- HVX sum of the (masked) exp bytes ---- */
        HVX_Vector accW = Q6_Vuw_vrmpyacc_VuwVubRub(Q6_V_vzero(), eu.v, 0x01010101u);
        int32_t S = hvec_hsum_w(accW);

        /* ---- scalar epilogue: recip lookup + final multiply-shift-clamp ---- */
        int32_t sidx = S >> 6;
        if (sidx < 0) sidx = 0;
        if (sidx > 255) sidx = 255;
        int32_t recip = (int32_t)recip_lut[sidx];

        for (int j = 0; j < C; j++) {
            int32_t ej = (int32_t)eu.b[j];
            int32_t v  = (ej * recip + 32768) >> 16;
            if (v < 0) v = 0;
            if (v > 255) v = 255;
            orow[j] = (uint8_t)v;
        }
    }
}
