/* Near-miss: indexes B as if transposed (B[j*K+k]) instead of B[k*N+j]. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, int32_t *C, int M, int N, int K){
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        int32_t acc=0;
        for (int k=0;k<K;k++) acc += (int32_t)A[i*K+k]*(int32_t)B[j*K+k];
        C[i*N+j]=acc;
    }
}
