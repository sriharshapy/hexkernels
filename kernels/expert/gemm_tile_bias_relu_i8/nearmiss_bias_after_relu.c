/* Near-miss: applies relu BEFORE adding bias -- computes
 *   clamp(max(acc,0) + bias[j], 0, 127)
 * instead of the correct
 *   clamp(max(acc + bias[j], 0), 0, 127).
 * Diverges whenever bias[j] is negative and acc is small-positive, or bias[j]
 * is positive and acc is very negative -- e.g. the harness's (0,0) cell has
 * acc=-80, bias[0]=90: correct=clamp(-80+90,0,127)=10, this near-miss gives
 * clamp(max(-80,0)+90,0,127)=90. */
#include <stdint.h>
void candidate_kernel(const int8_t *A, const int8_t *B, const int32_t *bias, int8_t *C,
                      int M, int N, int K){
    for (int i=0;i<M;i++) for (int j=0;j<N;j++){
        int32_t acc=0;
        for (int k=0;k<K;k++) acc += (int32_t)A[i*K+k]*(int32_t)B[j*K+k];
        int32_t relu = acc > 0 ? acc : 0;      /* WRONG: relu applied before bias */
        int32_t sum = relu + bias[j];
        int32_t v = sum > 127 ? 127 : (sum < 0 ? 0 : sum);
        C[i*N+j] = (int8_t)v;
    }
}
