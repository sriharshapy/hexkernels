#include <stdint.h>
void candidate_kernel(const int8_t *a, const int8_t *b, int32_t *C,
                      int M, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i*N+j] = (int32_t)a[i] * (int32_t)b[j];
}
