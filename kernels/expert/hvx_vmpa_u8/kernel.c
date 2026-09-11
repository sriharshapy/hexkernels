/* expert: HVX dual weighted-sum via Q6_Wh_vmpa_WubWub.
 * Empirically-pinned semantics of Q6_Wh_vmpa_WubWub(Vuu={hi=B,lo=A},
 * Vvv={hi=WB,lo=WA}): Q6_V_lo_W(result)[j] = A[2j]*WA[2j] + B[2j]*WB[2j];
 * Q6_V_hi_W(result)[j] = A[2j+1]*WA[2j+1] + B[2j+1]*WB[2j+1] (j=0..63).
 * i.e. the pair-of-pairs instruction computes the SAME per-lane formula
 * out[i]=A[i]*WA[i]+B[i]*WB[i] for i=0..127, just physically split by
 * index parity into two output vectors -- so the vector part is a plain
 * elementwise dual-MAC; only the final store needs a parity de-interleave. */
#include <stdint.h>
#include <hexagon_types.h>
#include <hexagon_protos.h>

void candidate_kernel(const uint8_t *a, const uint8_t *b,
                      const uint8_t *wa, const uint8_t *wb,
                      int16_t *out, int n) {
    int i = 0;
    for (; i + 128 <= n; i += 128) {
        HVX_Vector va  = *(const HVX_Vector *)(a  + i);
        HVX_Vector vb  = *(const HVX_Vector *)(b  + i);
        HVX_Vector vwa = *(const HVX_Vector *)(wa + i);
        HVX_Vector vwb = *(const HVX_Vector *)(wb + i);
        HVX_VectorPair Vuu = Q6_W_vcombine_VV(vb, va);
        HVX_VectorPair Vvv = Q6_W_vcombine_VV(vwb, vwa);
        HVX_VectorPair Rp = Q6_Wh_vmpa_WubWub(Vuu, Vvv);
        int16_t lo[64] __attribute__((aligned(128)));
        int16_t hi[64] __attribute__((aligned(128)));
        *(HVX_Vector *)lo = Q6_V_lo_W(Rp);
        *(HVX_Vector *)hi = Q6_V_hi_W(Rp);
        for (int j = 0; j < 64; j++) {
            out[i + 2*j]     = lo[j];
            out[i + 2*j + 1] = hi[j];
        }
    }
    for (; i < n; i++)
        out[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i]);
}
