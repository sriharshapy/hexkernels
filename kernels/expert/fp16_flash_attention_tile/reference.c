/* DENOMINATOR baseline -- plain scalar C, no HVX. Implements the STANDARD
 * (non-flash, full-softmax) attention formula directly; mathematically
 * equal to the flash recurrence (that's the point of flash-attention -- it's
 * an algorithmic reformulation of the SAME function, not a different one). */
#include "kernel_api.h"
#include "harness_common.h"
#include <math.h>

void candidate_kernel(const hvx_hf *Q, const hvx_hf *K, const hvx_hf *V,
                      hvx_hf *O, int SQ, int SK, int DH) {
    for (int i = 0; i < SQ; i++) {
        float sc[64];
        float rowmax = -1e30f;
        for (int j = 0; j < SK; j++) {
            float acc = 0.0f;
            for (int d = 0; d < DH; d++) acc += (float)Q[i*DH+d] * (float)K[j*DH+d];
            sc[j] = acc * FLASH_SCALE;
            if (sc[j] > rowmax) rowmax = sc[j];
        }
        float e[64], rowsum = 0.0f;
        for (int j = 0; j < SK; j++) { e[j] = expf(sc[j] - rowmax); rowsum += e[j]; }
        for (int d = 0; d < DH; d++) {
            float acc = 0.0f;
            for (int j = 0; j < SK; j++) acc += e[j] * (float)V[j*DH+d];
            O[i*DH+d] = (hvx_hf)(acc / rowsum);
        }
    }
}
