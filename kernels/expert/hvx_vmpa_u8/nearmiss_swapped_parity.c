/* NEAR-MISS: swaps the lo/hi parity de-interleave (writes hi to the EVEN
 * output slots and lo to the ODD slots). Compiles, and happens to be
 * correct only when a[i]==b[i] and wa[i]==wb[i] for the swapped pair, but
 * fails bit-exact on generic random inputs where even/odd terms differ. */
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
            out[i + 2*j]     = hi[j];   /* WRONG: swapped */
            out[i + 2*j + 1] = lo[j];   /* WRONG: swapped */
        }
    }
    for (; i < n; i++)
        out[i] = (int16_t)((int)a[i]*(int)wa[i] + (int)b[i]*(int)wb[i]);
}
