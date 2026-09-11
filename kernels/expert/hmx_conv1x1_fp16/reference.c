/* Scalar fp16 1x1 conv (= In * W^T, tolerance-correct) -- DENOMINATOR
 * baseline. Plain float32 accumulation cast to __fp16, matching the
 * harness's own scalar reference computation exactly. No HVX, no HMX. */
#include "kernel_api.h"

void candidate_kernel(const hvx_hf *In, const hvx_hf *W, hvx_hf *out, int n) {
    for (int p = 0; p < n; p++) {
        for (int co = 0; co < n; co++) {
            float acc = 0.f;
            for (int ci = 0; ci < n; ci++) acc += (float)In[p*n+ci] * (float)W[co*n+ci];
            out[p*n+co] = (hvx_hf)acc;
        }
    }
}
