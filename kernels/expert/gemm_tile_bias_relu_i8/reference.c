#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias, int8_t *C,
                      int M, int N, int K){
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        int32_t acc=0;
        for (int k=0;k<K;k++) acc += (int32_t)A[i*K+k]*(int32_t)B[j*K+k];
        acc += bias[j];
        int32_t relu = acc > 0 ? acc : 0;
        int32_t v = relu > 127 ? 127 : relu;
        C[i*N+j] = (int8_t)v;
    }
}
