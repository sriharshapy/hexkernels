/*
 * i8_lut_tanh — HVX 256-entry LUT, full unsigned index (uint8_t)in[i]
 *
 * Strategy:
 *   The index is the raw unsigned reinterpretation of the int8 input:
 *   (uint8_t)in[i] in [0,255] -- no clamping or bias needed.
 *
 *   1. Preprocess the 256-entry LUT into two shuffled 128-byte halves using
 *      Q6_Vb_vshuff_Vb (bilateral SDK vlut32 segment layout).
 *   2. 8 passes of Q6_Vb_vlut32 / Q6_Vb_vlut32or with segments 0-7 cover all
 *      256 entries. Segments 0-3 address sTab0 (lut[0..127]), segments 4-7
 *      address sTab1 (lut[128..255]).
 *   3. Handle tail with Q6_Q_vsetq2_R predicated store.
 *
 * N=1024 is a multiple of 128, so tail=0 in practice, but handled generically.
 */

#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const int8_t *in, int8_t *out, int n, const int8_t *lut)
{
    /*
     * Preprocess LUT: vshuff rearranges each 128-byte half into the segment
     * layout that vlut32 expects (each 32-entry segment reordered for SIMD lookup).
     */
    HVX_Vector sTab0 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut));         /* lut[0..127]   */
    HVX_Vector sTab1 = Q6_Vb_vshuff_Vb(*(const HVX_Vector *)(lut + 128));   /* lut[128..255] */

    int vlen = 128;
    int nvec = n / vlen;
    int tail = n % vlen;

    const int8_t *src = in;
    int8_t       *dst = out;

    for (int v = 0; v < nvec; v++) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        /*
         * 8-pass 256-entry LUT lookup.
         * vlut32 selects 32 entries per pass based on index[6:5] (segment field).
         * Segments 0-3 cover lut[0..127] (sTab0), segments 4-7 cover lut[128..255] (sTab1).
         */
        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR   (vidx, sTab0, 0);
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

    /* Tail (n=1024 → tail=0, but handled generically). */
    if (tail > 0) {
        HVX_Vector vidx = *(const HVX_Vector *)src;

        HVX_Vector res;
        res = Q6_Vb_vlut32_VbVbR   (vidx, sTab0, 0);
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
