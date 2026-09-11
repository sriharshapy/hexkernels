/* PLAIN SCALAR fp16 32x32x32 matmul + tanh -- the DENOMINATOR baseline. No
 * HVX intrinsics, no vector types anywhere. */
#include "kernel_api.h"
#include <math.h>

void candidate_kernel(const hvx_hf *A, const hvx_hf *B, hvx_hf *out, int n, int k_dim) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int k = 0; k < k_dim; k++)
                acc += (float)A[i*k_dim + k] * (float)B[k*n + j];
            hvx_hf m = (hvx_hf)acc;
            out[i*n + j] = (hvx_hf)tanhf((float)m);
        }
    }
}
