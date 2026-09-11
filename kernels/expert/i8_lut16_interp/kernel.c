/*
 * i8_lut16_interp — HVX kernel
 *
 * Strategy: precompute a full 256-entry expanded LUT from the runtime 17-entry
 * lut[], then apply it with the standard 8-pass Q6_Vb_vlut32/vlut32or idiom.
 *
 * For each possible input byte b in [0,255]:
 *   hi = b >> 4        (table index, 0..15)
 *   lo = b & 0xF       (interpolation fraction, 0..15)
 *   expanded[b] = (int8_t)(lut[hi] + (((int16_t)(lut[hi+1]-lut[hi]) * lo) >> 4))
 *
 * The shift is TRUNCATING (arithmetic right shift of int16_t).
 * This exactly matches baseline.c.
 *
 * After expansion, use the 256-entry HVX vlut32 pattern (8 passes with vshuff
 * on each 128-byte half) to map all 128 elements per iteration.
 *
 * N=1000 → 7 full vectors (7*128=896), tail=104.
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut)
{
    /* Build 256-entry expanded table matching the baseline's truncating formula. */
    int8_t expanded[256] __attribute__((aligned(128)));
    for (int b = 0; b < 256; b++) {
        int     hi    = b >> 4;
        int     lo    = b & 0xF;
        int16_t base  = lut[hi];
        int16_t delta = (int16_t)(lut[hi + 1] - lut[hi]);
        expanded[b]   = (int8_t)(base + ((delta * lo) >> 4));
    }

    /* Preprocess into vlut32 segment layout (bilateral SDK idiom). */
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(expanded));       /* [0..127]   */
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(expanded + 128)); /* [128..255] */

    int vlen = 128;
    int nvec = n / vlen;
    int tail = n % vlen;

    const int8_t *src = in;
    int8_t       *dst = out;

    for (int v = 0; v < nvec; v++) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);

        *(HVX_Vector *)dst = res;
        src += vlen;
        dst += vlen;
    }

    /* Tail: n=1000 → tail=104 bytes. */
    if (tail > 0) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR    (vidx, sTab0, 0);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 1);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 2);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab0, 3);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 4);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 5);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 6);
        res = Q6_Vb_vlut32or_VbVbVbR(res, vidx, sTab1, 7);

        HVX_VectorPred mask = Q6_Q_vsetq2_R(tail);
        Q6_vmem_QRIV(mask, (HVX_Vector *)dst, res);
    }
}
